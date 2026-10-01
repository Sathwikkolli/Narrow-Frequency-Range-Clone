#!/bin/bash
#SBATCH --job-name=subband
#SBATCH --account=hafiz1
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8g
#SBATCH --time=02:00:00
#SBATCH --array=0-19
#SBATCH --output=logs/subband_%A_%a.log

# --- low-pass / high-pass copies of ASVspoof 2019 LA eval, for the sub-band sweep ---
#
# CPU only. One array task per shard; each task builds every condition for its
# shard. See eval/build_subband.py for what the conditions are and why.
#
# SUBMIT (from the Clone repo root):
#   mkdir -p logs
#   LIMIT=50 sbatch --array=0 eval/build_subband.sbatch          # smoke test
#   sbatch eval/build_subband.sbatch                             # all ten conditions
#   COND="LP3400 HP300" sbatch eval/build_subband.sbatch         # a subset
#
# DISK. Each condition is ~13 GB / ~72k files on turbo; the default ten are ~130 GB.
#
# HEADROOM must match what F0 was built with (-6), since F0 is the control.

set -u

CLONE_DIR=${CLONE_DIR:-${SLURM_SUBMIT_DIR:-$HOME/Narrow-Frequency-Range-Clone}}
NFR_DIR=${NFR_DIR:-$HOME/Narrow-Frequency-Range}
SRC=${SRC:-/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA}
OUT=${OUT:-/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019_NFR}
COND=${COND:-all}
SUBSET=${SUBSET:-eval}
NSHARDS=${NSHARDS:-20}
LIMIT=${LIMIT:-0}
HEADROOM=${HEADROOM:--6}

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-wmcompare}"

SHARD=${SLURM_ARRAY_TASK_ID:-0}

[ -d "$SRC" ] || { echo "SRC not found: $SRC"; exit 1; }
[ -f "$CLONE_DIR/eval/build_subband.py" ] || { echo "build_subband.py not found under $CLONE_DIR"; exit 1; }

echo "clone    $CLONE_DIR  ($(git -C "$CLONE_DIR" rev-parse --short HEAD 2>/dev/null))"
echo "nfr      $NFR_DIR  ($(git -C "$NFR_DIR" rev-parse --short HEAD 2>/dev/null))"
echo "src      $SRC"
echo "out      $OUT"
echo "cond     $COND   subset $SUBSET   shard $SHARD/$NSHARDS   headroom ${HEADROOM} dB"
echo "python   $(which python)"

python "$CLONE_DIR/eval/build_subband.py" \
    --nfr-dir "$NFR_DIR" \
    --cond "$COND" \
    --src "$SRC" \
    --out "$OUT" \
    --subset "$SUBSET" \
    --shard "$SHARD" \
    --nshards "$NSHARDS" \
    --headroom-db "$HEADROOM" \
    --limit "$LIMIT"
