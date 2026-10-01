#!/bin/bash
# Queue the low-pass / high-pass sweep for the five harness models on
# ASVspoof2019 LA and In-the-Wild, in one go. Login node, from the Clone root:
#
#   bash eval/submit_subband.sh            # check everything, then submit
#   DRY=1 bash eval/submit_subband.sh      # check and print, submit nothing
#   DATASETS=itw bash eval/submit_subband.sh
#
# What it queues:
#   ASVspoof2019   score rawnet2 aasist nes2net tcm sls      -> runs/subband
#   In-the-Wild    build the LP/HP audio into $ITW_NFR
#                  score the same five, once the build is ok  -> runs/subband_itw
#
# LCNN is not here: it runs through its own tooling (eval/run_lcnn.sh).
#
# Each model is launched with the conda env and checkpoint its F-condition runs
# used. That matters: run_suite.sbatch's defaults are wmcompare and "the first
# .pth in the folder", which is neither the env the XLS-R models need nor
# necessarily the checkpoint the published numbers came from.
#
# Finished score shards are skipped, so this is safe to re-run after a failure.
# In the morning: bash eval/finish_subband.sh

set -eu
cd "$(dirname "${BASH_SOURCE[0]}")/.."
CLONE_DIR=$PWD

D=/nfs/turbo/umd-hafiz/issf_server_data
SCR=${SCR:-/scratch/hafiz_root/hafiz1/$USER}
MODELS=${MODELS:-$HOME/models}
PART=${PART:-gpu,spgpu}
DATASETS=${DATASETS:-"asv itw"}
ADAPTERS=${ADAPTERS:-"rawnet2 aasist nes2net tcm sls"}
CONDS=${CONDS:-"LP2000 LP3400 LP4000 LP5000 LP6000 LP7000 HP100 HP300 HP500 HP1000"}
DRY=${DRY:-0}

ASV_NFR=${ASV_NFR:-$D/AsvSpoofData_2019_NFR}
ITW_ROOT=${ITW_ROOT:-$D/ITW_16k}
ITW_NFR=${ITW_NFR:-$D/ITW_NFR}
ITW_LABELS=${ITW_LABELS:-$ITW_ROOT/protocol/ITW.cm.eval.trl.txt}

env_of() {
    case "$1" in
      rawnet2) echo wmcompare ;;
      sls)     echo "$SCR/envs/sls" ;;
      *)       echo ssl_spoof ;;
    esac
}
# empty means "let run_suite.sbatch pick", which is what those runs did
ckpt_of() {
    case "$1" in
      aasist) echo "$MODELS/aasist_ckpt/LA_model.pth" ;;
      sls)    echo "$MODELS/sls_ckpt/MMpaper_model.pth" ;;
      *)      echo "" ;;
    esac
}

# ---------------------------------------------------------------- preflight
bad=0
need() { [ -e "$1" ] || { echo "MISSING  $1   ($2)"; bad=1; }; }

for a in $ADAPTERS; do
    e=$(env_of "$a"); c=$(ckpt_of "$a")
    case "$e" in /*) need "$e" "conda env for $a" ;;
                  *) need "$HOME/.conda/envs/$e" "conda env for $a" ;;
    esac
    [ -z "$c" ] || need "$c" "checkpoint for $a"
done
for ds in $DATASETS; do
    case "$ds" in
      asv) for c in $CONDS; do need "$ASV_NFR/ASVspoof2019_LA_eval/$c/flac" "ASV $c audio"; done
           need "$ASV_NFR/ASVspoof2019_LA_eval/F0/flac" "ASV F0 control" ;;
      itw) need "$ITW_ROOT/ASVspoof2019_LA_eval/flac" "ITW source audio"
           need "$ITW_LABELS" "ITW protocol"
           need "$ITW_NFR/ASVspoof2019_LA_eval/F0/flac" "ITW F0 control"
           need "${NFR_DIR:-$HOME/Narrow-Frequency-Range}/nfr_dataset/build_shard.py" "pipeline repo" ;;
      *)   echo "unknown dataset '$ds' (asv, itw)"; bad=1 ;;
    esac
done

# Scores written by a run this script did not launch may be from the wrong env or
# checkpoint, and the resume check would silently keep them.
for dir in runs/subband runs/subband_itw; do
    if ls "$dir"/scores/*.csv >/dev/null 2>&1 && [ ! -f "$dir/.submit_subband" ]; then
        echo "STALE    $dir/scores holds $(ls "$dir"/scores/*.csv | wc -l) score file(s) from an earlier"
        echo "         submission. Move the folder aside first:  mv $dir ${dir}_old"
        bad=1
    fi
done
[ "$bad" = 0 ] || { echo; echo "nothing submitted"; exit 1; }
echo "preflight ok"

# ---------------------------------------------------------------- submit
mkdir -p logs
JOBS=runs/subband_jobs.txt
run() {  # run <label> <sbatch args...>  -> job id on stdout
    local label=$1; shift
    if [ "$DRY" = 1 ]; then echo "DRY  $label: sbatch $*" >&2; echo 0; return; fi
    local id; id=$(sbatch --parsable "$@"); id=${id%%;*}
    echo "$(date '+%F %T')  $id  $label" | tee -a "$JOBS" >&2
    echo "$id"
}

score() {  # score <dataset-label> <out-dir> <adapter> [extra sbatch args...]
    local ds=$1 out=$2 a=$3; shift 3
    mkdir -p "$out"; [ "$DRY" = 1 ] || touch "$out/.submit_subband"
    ADAPTER=$a CONDA_ENV=$(env_of "$a") CKPT=$(ckpt_of "$a") CONDITIONS="$CONDS" \
    OUT_DIR=$CLONE_DIR/$out CLONE_DIR=$CLONE_DIR MODELS=$MODELS \
        run "$ds score $a" --partition="$PART" --mem=24g "$@" eval/run_suite.sbatch >/dev/null
}

for ds in $DATASETS; do
    case "$ds" in
      asv)
        for a in $ADAPTERS; do
            NFR_ROOT=$ASV_NFR score asv runs/subband "$a"
        done ;;
      itw)
        # same headroom as ITW's F0 control (-6), which build_subband.sbatch defaults to
        BUILD=$(SRC=$ITW_ROOT OUT=$ITW_NFR COND="$CONDS" CLONE_DIR=$CLONE_DIR \
                run "itw build" eval/build_subband.sbatch)
        DEP=(); [ "$DRY" = 1 ] || DEP=(--dependency="afterok:$BUILD")
        for a in $ADAPTERS; do
            RAW_ROOT=$ITW_ROOT NFR_ROOT=$ITW_NFR LABELS=$ITW_LABELS \
                score itw runs/subband_itw "$a" ${DEP[@]+"${DEP[@]}"}
        done ;;
    esac
done

echo
[ "$DRY" = 1 ] && echo "dry run: nothing submitted" || echo "job ids appended to $JOBS;  watch with: squeue --me"
