#!/bin/bash
#SBATCH --job-name=nfr_trump_5v5
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4g
#SBATCH --time=00:20:00
#SBATCH --account=hafiz1
#SBATCH --output=nfr_trump_5v5_%j.log

# --- 5 real + 5 fake Donald Trump clips -> telephone narrowband (IRS8, 8 kHz, -26 dBov) ---
#
#   1. link the clips in samples/trump_5real_5fake.csv from DATA_ROOT into RUN_DIR/input
#   2. run Narrow-Frequency-Range/stl_pipeline.py on them  ->  RUN_DIR/output
#
# One-time, on a login node (compute nodes may not reach GitHub):
#   bash $HOME/Narrow-Frequency-Range/setup_stl.sh
# Submit:
#   sbatch $HOME/Narrow-Frequency-Range-Clone/run_trump_samples.sbatch

source "${CLONE_DIR:-$HOME/Narrow-Frequency-Range-Clone}/config.sh"

eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

for t in filter sv56demo; do
    [ -x "$STL_BIN/$t" ] || { echo "STL tool $STL_BIN/$t missing: run bash $NFR_DIR/setup_stl.sh first"; exit 1; }
done

echo "pipeline repo: $NFR_DIR ($(git -C "$NFR_DIR" rev-parse --short HEAD 2>/dev/null))"
echo "data root:     $DATA_ROOT"
echo "run dir:       $RUN_DIR"
echo "python:        $(which python)"

echo; echo "===== 1/2 prepare samples ====="
python "$CLONE_DIR/prepare_samples.py" --data-root "$DATA_ROOT" --samples "$SAMPLES_CSV" --out "$RUN_DIR/input" || exit 1

echo; echo "===== 2/2 narrowband pipeline ====="
python "$NFR_DIR/stl_pipeline.py" --stl-bin "$STL_BIN" --input "$RUN_DIR/input" --output "$RUN_DIR/output"
