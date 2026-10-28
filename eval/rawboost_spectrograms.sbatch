#!/bin/bash
#SBATCH --job-name=rb_spec
#SBATCH --account=hafiz1
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8g
#SBATCH --time=00:30:00
#SBATCH --output=logs/rb_spec_%j.log

# --- spectrograms of 5 C1 (ITU) clips after RawBoost algos 1-5; see eval/rawboost_spectrograms.py
#
# SUBMIT (from the Clone repo root):
#   mkdir -p logs && sbatch eval/rawboost_spectrograms.sbatch
#   N=3 ALGOS="3 4 5" sbatch eval/rawboost_spectrograms.sbatch
#   # one model, every panel as its own image (W2V2-AASIST = algo 5):
#   ALGOS=5 MODEL=W2V2-AASIST OUT=$PWD/runs/rawboost_spec_aasist EXTRA=--panels \
#       sbatch eval/rawboost_spectrograms.sbatch
#
# CPU only, a few minutes. Needs the env the official runs trained in (ssl_spoof:
# librosa, soxr, torch for the harness import) plus matplotlib.

set -u

CLONE_DIR=${CLONE_DIR:-${SLURM_SUBMIT_DIR:-$HOME/Narrow-Frequency-Range-Clone}}
OUT=${OUT:-$CLONE_DIR/runs/rawboost_spec}
REPO=${REPO:-$HOME/models/tcm_repo}
N=${N:-5}
ALGOS=${ALGOS:-1 2 3 4 5}
SEED=${SEED:-1234}
SUBSET=${SUBSET:-train}
MODEL=${MODEL:-}
EXTRA=${EXTRA:-}

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-ssl_spoof}"

echo "clone    $CLONE_DIR  ($(git -C "$CLONE_DIR" rev-parse --short HEAD 2>/dev/null))"
echo "repo     $REPO"
echo "out      $OUT"
echo "python   $(which python)"

# shellcheck disable=SC2086  # ALGOS and EXTRA are lists
python "$CLONE_DIR/eval/rawboost_spectrograms.py" \
    --repo "$REPO" --out "$OUT" --n "$N" --algos $ALGOS \
    --seed "$SEED" --subset "$SUBSET" --model "$MODEL" $EXTRA
