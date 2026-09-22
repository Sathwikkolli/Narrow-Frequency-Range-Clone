#!/bin/bash
# Paths on Great Lakes. Sourced by the .sbatch files; override any of them by
# exporting it before sbatch, e.g.  DATA_ROOT=/other/path sbatch run_trump_samples.sbatch

# the pipeline repo (issflab/Narrow-Frequency-Range) and this repo
export NFR_DIR=${NFR_DIR:-$HOME/Narrow-Frequency-Range}
export CLONE_DIR=${CLONE_DIR:-$HOME/Narrow-Frequency-Range-Clone}

# compiled STL programs (built once by $NFR_DIR/setup_stl.sh)
export STL_BIN=${STL_BIN:-$NFR_DIR/STL/build/bin}

# Famous Figures dataset, Donald Trump. Layout: {Speaker}/{Source}/{AudioName}.wav,
# real clips under "-" (Great Lakes) or "Bonafide" (lab server copy).
#   Great Lakes (Deep_SVDD, OneClass, Spectral_Features): the default below
#   ISSF lab server (spoof_SUPERB, ExpertASD):            /data/Data/famousfigures/Donald_Trump
export DATA_ROOT=${DATA_ROOT:-/nfs/turbo/umd-hafiz/issf_server_data/famousfigures/Donald_Trump}

# which clips to take, and where inputs/outputs of a run go
export SAMPLES_CSV=${SAMPLES_CSV:-$CLONE_DIR/samples/trump_5real_5fake.csv}
export RUN_DIR=${RUN_DIR:-$CLONE_DIR/runs/trump_5real_5fake}

# python env with numpy + scipy
export CONDA_ENV=${CONDA_ENV:-wmcompare}
