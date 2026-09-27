#!/bin/bash
#SBATCH --job-name=dfarena
#SBATCH --account=hafiz1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=48g
#SBATCH --time=01:30:00
# Partition note: on Great Lakes, "gpu" is V100 16 GB (no TF32 -- Volta predates it),
# "spgpu" is A40 48 GB (TF32 and bf16 work, ~2-3x faster, and less contested).
# Override on the command line, e.g.  sbatch --partition=spgpu eval/run_df_arena.sbatch
#SBATCH --array=0-19
#SBATCH --output=logs/dfarena_%A_%a.log

# --- Speech-Arena-2025/DF_Arena_1B_V_1 on narrowband ASVspoof2019 LA ---
#
# One array task per manifest shard; each task scores every condition for that shard,
# so the 4.6 GB model is loaded once per task rather than once per condition. Per-task
# outputs are merged afterwards by eval/merge.sh.
#
# ONE-TIME, on a login node (compute nodes are offline):
#   export HF_HOME=/scratch/hafiz_root/hafiz1/$USER/hf
#   python -c "from transformers import AutoModel; \
#              AutoModel.from_pretrained('Speech-Arena-2025/DF_Arena_1B_V_1', trust_remote_code=True)"
#
# SUBMIT:
#   mkdir -p logs
#   sbatch eval/run_df_arena.sbatch                          # all 20 shards, 3 conditions
#   CONDITIONS=P0 sbatch --array=0 eval/run_df_arena.sbatch   # one shard, one condition
#   LIMIT=200 CONDITIONS=P0 sbatch --array=0 eval/run_df_arena.sbatch    # smoke test

set -u

CLONE_DIR=${CLONE_DIR:-$HOME/Narrow-Frequency-Range-Clone}
NB_ROOT=${NB_ROOT:-/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019_NB}
LABELS=${LABELS:-$HOME/asvspoof_protocols/ASVspoof2019.LA.cm.eval.trl.txt}
SPLIT=${SPLIT:-eval}
CONDITIONS=${CONDITIONS:-"P0 B1 C1"}
OUT_DIR=${OUT_DIR:-$CLONE_DIR/runs/dfarena_$SPLIT}
MODEL=${MODEL:-Speech-Arena-2025/DF_Arena_1B_V_1}
BATCH=${BATCH:-16}                # I/O prefetch group, not a model batch: the model
                                  # scores one clip per forward (backbone.py adds the
                                  # batch axis itself). Does not affect GPU memory.
PRECISION=${PRECISION:-fp32}      # fp32 for the P0 validation; fp16 once it is confirmed
                                  # (tf32 is a no-op on V100 -- Ampere and later only)
LIMIT=${LIMIT:-0}
EMBED=${EMBED:-1}                 # 0 to skip embeddings (smaller output, no UMAP)

export HF_HOME=${HF_HOME:-/scratch/hafiz_root/hafiz1/$USER/hf}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-wmcompare}"

SHARD=$(printf "%04d" "${SLURM_ARRAY_TASK_ID:-0}")

[ -f "$LABELS" ] || { echo "protocol not found: $LABELS"; exit 1; }
[ -d "$NB_ROOT" ] || { echo "dataset root not found: $NB_ROOT"; exit 1; }

echo "model      $MODEL  ($PRECISION, batch $BATCH)"
echo "split      $SPLIT   shard $SHARD"
echo "conditions $CONDITIONS"
echo "labels     $LABELS"
echo "out        $OUT_DIR"
echo "python     $(which python)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

mkdir -p "$OUT_DIR/scores" "$OUT_DIR/embeddings"
rc=0
for cond in $CONDITIONS; do
    man="$NB_ROOT/manifest/${cond}_${SPLIT}_${SHARD}.csv"
    if [ ! -f "$man" ]; then
        echo "skipping $cond: no shard $man"
        continue
    fi
    echo; echo "===== $cond shard $SHARD ====="
    emb_arg=()
    [ "$EMBED" = "1" ] && emb_arg=(--save-embeddings "$OUT_DIR/embeddings/${cond}_${SHARD}.npz")
    python "$CLONE_DIR/eval/score_df_arena.py" \
        --manifest "$man" \
        --labels "$LABELS" \
        --nb-root "$NB_ROOT" \
        --out "$OUT_DIR/scores/${cond}_${SHARD}.csv" \
        --model "$MODEL" \
        --batch-size "$BATCH" \
        --workers "${SLURM_CPUS_PER_TASK:-8}" \
        --precision "$PRECISION" \
        --limit "$LIMIT" \
        "${emb_arg[@]}" || rc=1
done

exit $rc
