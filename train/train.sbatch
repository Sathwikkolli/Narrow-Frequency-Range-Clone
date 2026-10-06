#!/bin/bash
#SBATCH --account=hafiz1
#SBATCH --partition=spgpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus=1
#SBATCH --mem=48g
#SBATCH --time=2-00:00:00

# --- one training run; submitted by train/submit.sh, not by hand ---
#
# submit.sh queues several copies of this job under one name with
# --dependency=singleton, so they run one after another. Each copy resumes from
# last.pth, and once the run has written DONE.json the remaining copies exit in
# seconds. That is how a run of unknown length fits in 2-day slots.
#
# spgpu (A40, 48 GB) only: fine-tuning XLS-R at the published batch size does not
# fit a 16 GB V100.
#
# Nothing heavy is written to home (8 GB free): checkpoints go to RUN_DIR and every
# cache to scratch. When the run finishes, best.pth and its records are copied to
# FINAL_DIR on turbo, which is not purged, and the copy is checksummed.

set -u
: "${MODEL:?}" "${DATA:?}" "${SEED:?}" "${RUN_DIR:?}" "${REPO:?}" "${DATA_ROOT:?}" \
  "${PROTOCOLS:?}" "${SCR:?}" "${CLONE_DIR:?}"
FINAL_DIR=${FINAL_DIR:-}
EXTRA=${EXTRA:-}

export TMPDIR=$SCR/itu_train/cache/tmp
export XDG_CACHE_HOME=$SCR/itu_train/cache
export TORCH_HOME=$SCR/itu_train/cache/torch
export HF_HOME=$SCR/itu_train/cache/hf
export PIP_CACHE_DIR=$SCR/itu_train/cache/pip
export PYTHONUNBUFFERED=1
mkdir -p "$TMPDIR" "$RUN_DIR"

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-ssl_spoof}"

echo "run        $(basename "$RUN_DIR")   job ${SLURM_JOB_ID:-}   $(hostname)"
echo "model      $MODEL   data $DATA   seed $SEED"
echo "run dir    $RUN_DIR"
echo "final dir  ${FINAL_DIR:-<none, smoke test>}"
echo "python     $(which python)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null

# shellcheck disable=SC2086  # EXTRA is a list of flags
python "$CLONE_DIR/train/train.py" \
    --model "$MODEL" --data "$DATA" --seed "$SEED" \
    --run-dir "$RUN_DIR" --repo "$REPO" \
    --data-root "$DATA_ROOT" --protocols "$PROTOCOLS" \
    --workers "${SLURM_CPUS_PER_TASK:-8}" \
    $EXTRA
rc=$?
[ $rc -eq 0 ] || { echo "train.py exited $rc"; exit $rc; }

# Not finished yet (time limit next time round), or a smoke test: nothing to copy.
[ -f "$RUN_DIR/DONE.json" ] || exit 0
[ -n "$FINAL_DIR" ] || exit 0

mkdir -p "$FINAL_DIR"
want=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['best_sha256'])" "$RUN_DIR/DONE.json")
if [ -f "$FINAL_DIR/best.pth" ] && [ "$(sha256sum "$FINAL_DIR/best.pth" | cut -d' ' -f1)" = "$want" ]; then
    echo "final copy already in place: $FINAL_DIR"
    exit 0
fi
for f in best.pth config.json train_log.csv dev_scores_best.csv DONE.json starts.log; do
    [ -f "$RUN_DIR/$f" ] && cp "$RUN_DIR/$f" "$FINAL_DIR/"
done
got=$(sha256sum "$FINAL_DIR/best.pth" | cut -d' ' -f1)
if [ "$got" != "$want" ]; then
    echo "CHECKSUM MISMATCH copying best.pth to $FINAL_DIR ($got != $want)"
    exit 1
fi
(cd "$FINAL_DIR" && sha256sum best.pth > SHA256SUMS)
echo "copied to $FINAL_DIR (sha256 ok)"
