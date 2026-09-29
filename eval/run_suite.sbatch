#!/bin/bash
#SBATCH --job-name=nfrsuite
#SBATCH --account=hafiz1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=32g
#SBATCH --time=04:00:00
#SBATCH --array=0-19
#SBATCH --output=logs/suite_%A_%a.log

# --- one detector over every condition, sharded ---
#
# Each array task takes one shard of the trial list and scores EVERY condition
# for it, so the model is loaded once per task rather than once per condition.
# Inference needs 2-4 GB, so the V100 "gpu" partition is fine; "spgpu" (A40) is
# 2-3x faster and less contested when you want the results sooner:
#
#   ADAPTER=rawnet2 sbatch --partition=spgpu eval/run_suite.sbatch
#
# SUBMIT:
#   mkdir -p logs
#   ADAPTER=rawnet2 sbatch eval/run_suite.sbatch              # all 10 conditions
#   ADAPTER=tcm CONDITIONS="RAW P0 B1" sbatch eval/run_suite.sbatch
#   ADAPTER=rawnet2 LIMIT=200 sbatch --array=0 eval/run_suite.sbatch   # smoke test
#
# TCM: pass CKPT explicitly to pick a seed, e.g. CKPT=$MODELS/tcm_ckpt/best_0.pth
# TCM is fp32 only -- harness.py refuses fp16/bf16 for it (its SSLModel flips
# XLS-R into train() mode on a dtype change, silently, during scoring).
#
# LCNN is NOT run here: it is a project-NN-Pytorch-scripts project, not an
# importable module. See eval/adapters/lcnn_NOTES.md and eval/lcnn_to_csv.py.

set -u

CLONE_DIR=${CLONE_DIR:-$HOME/Narrow-Frequency-Range-Clone}
MODELS=${MODELS:-$HOME/models}

# corpora
RAW_ROOT=${RAW_ROOT:-/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA}
NB_ROOT=${NB_ROOT:-/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019_NB}
NFR_ROOT=${NFR_ROOT:-/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019_NFR}
LABELS=${LABELS:-$HOME/asvspoof_protocols/ASVspoof2019.LA.cm.eval.trl.txt}

ADAPTER=${ADAPTER:?set ADAPTER, e.g. ADAPTER=rawnet2}
CONDITIONS=${CONDITIONS:-"RAW P0 B1 C1 F0 F1 F2 F3 F4 F5"}
OUT_DIR=${OUT_DIR:-$CLONE_DIR/runs/suite}
NSHARDS=${NSHARDS:-20}
BATCH=${BATCH:-32}
WORKERS=${WORKERS:-8}
LIMIT=${LIMIT:-0}
PRECISION=${PRECISION:-fp32}
SAVE_EMB=${SAVE_EMB:-0}          # 1 to write .npz embeddings for the probe

SHARD=${SLURM_ARRAY_TASK_ID:-0}

# repo and checkpoint per adapter; override either from the command line
case "$ADAPTER" in
  rawnet2)
    REPO=${REPO:-$MODELS/asvspoof2021/LA/Baseline-RawNet2}
    CKPT=${CKPT:-$(ls "$MODELS"/rawnet2_ckpt/*.pth 2>/dev/null | head -1)} ;;
  aasist)
    REPO=${REPO:-$MODELS/aasist_repo}
    CKPT=${CKPT:-$(ls "$MODELS"/aasist_ckpt/*.pth 2>/dev/null | head -1)} ;;
  tcm)
    REPO=${REPO:-$MODELS/tcm_repo}
    CKPT=${CKPT:-$(ls "$MODELS"/tcm_ckpt/*.pth 2>/dev/null | head -1)} ;;
  sls)
    REPO=${REPO:-$MODELS/sls_repo}
    CKPT=${CKPT:-$(ls "$MODELS"/sls_ckpt/*.pth 2>/dev/null | head -1)} ;;
  nes2net)
    REPO=${REPO:-$MODELS/nes2net_repo}
    CKPT=${CKPT:-$(ls "$MODELS"/nes2net_ckpt/*.pth 2>/dev/null | head -1)} ;;
  *) echo "unknown ADAPTER '$ADAPTER'"; exit 1 ;;
esac

[ -n "${CKPT:-}" ] && [ -f "$CKPT" ] || {
    echo "no checkpoint for $ADAPTER (looked under $MODELS/${ADAPTER}_ckpt/)"
    echo "run eval/setup_models.sh, or pass CKPT=/path/to/model.pth"
    exit 1
}
[ -d "$REPO" ] || { echo "repo not found: $REPO"; exit 1; }
[ -f "$LABELS" ] || { echo "protocol not found: $LABELS"; exit 1; }

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-wmcompare}"

mkdir -p "$OUT_DIR/scores" "$OUT_DIR/emb"

echo "adapter    $ADAPTER"
echo "repo       $REPO"
echo "ckpt       $CKPT"
echo "conditions $CONDITIONS"
echo "shard      $SHARD/$NSHARDS   batch $BATCH   precision $PRECISION"
echo "python     $(which python)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null

fail=0
for COND in $CONDITIONS; do
    OUT="$OUT_DIR/scores/${ADAPTER}_${COND}_$(printf '%04d' "$SHARD").csv"
    if [ -s "$OUT" ]; then
        echo "--- $COND: $OUT exists, skipping ---"
        continue
    fi

    # RAW has no manifest: neither pipeline built it, so take the folder directly.
    case "$COND" in
      RAW) SRC_ARGS=(--audio-dir "$RAW_ROOT/ASVspoof2019_LA_eval/flac" --cond RAW) ;;
      P0|B1|C1)
           SRC_ARGS=(--manifest "$NB_ROOT/manifest/${COND}_eval_"*.csv --nb-root "$NB_ROOT") ;;
      F0|F1|F2|F3|F4|F5)
           SRC_ARGS=(--manifest "$NFR_ROOT/manifest/${COND}_eval_"*.csv --nb-root "$NFR_ROOT") ;;
      *)   echo "unknown condition '$COND'"; fail=1; continue ;;
    esac

    EMB_ARGS=()
    if [ "$SAVE_EMB" = "1" ]; then
        EMB_ARGS=(--save-embeddings "$OUT_DIR/emb/${ADAPTER}_${COND}_$(printf '%04d' "$SHARD").npz")
    fi

    echo
    echo "=== $ADAPTER / $COND / shard $SHARD ==="
    python "$CLONE_DIR/eval/harness.py" \
        --adapter "$ADAPTER" \
        --repo "$REPO" \
        --ckpt "$CKPT" \
        --labels "$LABELS" \
        "${SRC_ARGS[@]}" \
        --out "$OUT" \
        --shard "$SHARD" --nshards "$NSHARDS" \
        --batch-size "$BATCH" --workers "$WORKERS" \
        --precision "$PRECISION" \
        --limit "$LIMIT" \
        "${EMB_ARGS[@]}" || { echo "FAILED: $ADAPTER / $COND"; fail=1; }
done

exit $fail
