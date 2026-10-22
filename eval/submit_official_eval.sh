#!/bin/bash
# Score official-recipe checkpoints (train/official.py) on ASVspoof2019 LA eval and
# In-the-Wild, each under RAW, P0, B1 and C1. Login node, from anywhere:
#
#   bash eval/submit_official_eval.sh                       # every finished run below
#   bash eval/submit_official_eval.sh tcm_itu_official_s1   # just these runs
#   DRY=1 bash eval/submit_official_eval.sh                 # check paths, submit nothing
#   ITW_NB=/path/to/itw/nb bash eval/submit_official_eval.sh
#
# A run is scored only once its best.pth is on turbo ($CKPT_ROOT/<run>/best.pth,
# written when training finishes), so this can be re-run as runs complete: runs
# without a checkpoint are skipped, and eval/run_suite.sbatch itself skips any
# shard whose scores already exist.
#
# Test sets and conditions:
#   asv19   ASVspoof2019 LA eval, 71,237 trials   RAW (16 kHz original), P0 (+ P.56
#           level), B1 (+ IRS8 handset, 8 kHz), C1 (+ G.711, 8 kHz)
#   itw     In-the-Wild, 31,778 trials            the same four conditions
#
# Scores land in runs/official_eval/<testset>/<run>/; eval/finish_official_eval.sh
# merges them and writes results/official_eval/<testset>/<run>.json.
#
# SLS is not in the default list (set aside for now); pass sls_<data>_official_s1
# explicitly to score it -- it gets its own env and the V100 partition.

set -eu
cd "$(dirname "${BASH_SOURCE[0]}")/.."
CLONE_DIR=$PWD

T=/nfs/turbo/umd-hafiz/issf_server_data
SCR=${SCR:-/scratch/hafiz_root/hafiz1/$USER}
CKPT_ROOT=${CKPT_ROOT:-$T/official_ckpt}
DRY=${DRY:-0}
CONDS=${CONDS:-"RAW P0 B1 C1"}
TESTSETS=${TESTSETS:-"asv19 itw"}

# ASVspoof2019 LA eval: the same defaults run_suite.sbatch uses
ASV_RAW=${ASV_RAW:-$T/AsvSpoofData_2019/train/LA}
ASV_NB=${ASV_NB:-$T/AsvSpoofData_2019_NB}
ASV_LABELS=${ASV_LABELS:-$HOME/asvspoof_protocols/ASVspoof2019.LA.cm.eval.trl.txt}
# In-the-Wild
ITW_RAW=${ITW_RAW:-$T/ITW_16k}
ITW_NB=${ITW_NB:-$T/ITW_NB}
ITW_LABELS=${ITW_LABELS:-$ITW_RAW/protocol/ITW.cm.eval.trl.txt}

if [ $# -gt 0 ]; then
    RUNS="$*"
else
    RUNS=""
    for m in rawnet2 tcm aasist nes2net; do
        for d in wb itu; do RUNS="$RUNS ${m}_${d}_official_s1"; done
    done
fi

# --- check the corpora once, before anything is queued
fail=0
need() { [ -e "$1" ] || { echo "MISSING $2: $1"; fail=1; }; }
needglob() { compgen -G "$1" >/dev/null || { echo "MISSING $2: $1"; fail=1; }; }
for ts in $TESTSETS; do
    case "$ts" in
      asv19) need "$ASV_RAW/ASVspoof2019_LA_eval/flac" "2019 LA eval audio (RAW)"
             need "$ASV_LABELS" "2019 LA eval protocol"
             for c in $CONDS; do [ "$c" = RAW ] || needglob "$ASV_NB/manifest/${c}_eval_*.csv" "2019 LA $c manifest"; done ;;
      itw)   need "$ITW_RAW/ASVspoof2019_LA_eval/flac" "In-the-Wild audio (RAW)"
             need "$ITW_LABELS" "In-the-Wild protocol"
             for c in $CONDS; do [ "$c" = RAW ] || needglob "$ITW_NB/manifest/${c}_eval_*.csv" "In-the-Wild $c manifest (set ITW_NB)"; done ;;
      *)     echo "unknown test set '$ts' (asv19, itw)"; exit 1 ;;
    esac
done
if [ "$fail" != 0 ]; then
    echo
    echo "In-the-Wild's narrowband tree is the folder holding manifest/C1_eval_*.csv; find it with"
    echo "  ls -d $T/ITW*/manifest"
    exit 1
fi

mkdir -p "$CLONE_DIR/logs"
n=0
for run in $RUNS; do
    model=${run%%_*}
    ckpt=$CKPT_ROOT/$run/best.pth
    if [ ! -f "$ckpt" ]; then
        echo "skip  $run   (no $ckpt yet -- still training?)"
        continue
    fi
    case "$model" in
      rawnet2)          env=wmcompare;           part=gpu,spgpu; win=0 ;;
      aasist|tcm)       env=ssl_spoof;           part=gpu,spgpu; win=0 ;;
      # train/official.py trains Nes2Net on its data_utils_SSL cut, 64600
      nes2net)          env=ssl_spoof;           part=gpu,spgpu; win=64600 ;;
      # the SLS eval env's torch has no A40 kernels
      sls)              env=$SCR/envs/sls;       part=gpu;       win=0 ;;
      *) echo "unknown model in run name '$run'"; exit 1 ;;
    esac
    for ts in $TESTSETS; do
        if [ "$ts" = asv19 ]; then raw=$ASV_RAW; nb=$ASV_NB; labels=$ASV_LABELS
        else raw=$ITW_RAW; nb=$ITW_NB; labels=$ITW_LABELS; fi
        out=$CLONE_DIR/runs/official_eval/$ts/$run
        echo "queue $run on $ts [$CONDS] -> $out"
        n=$((n + 1))
        [ "$DRY" = 1 ] && continue
        ADAPTER=$model CONDA_ENV=$env CKPT=$ckpt WINDOW=$win \
        RAW_ROOT=$raw NB_ROOT=$nb LABELS=$labels CONDITIONS="$CONDS" OUT_DIR=$out \
        CLONE_DIR=$CLONE_DIR \
            sbatch --partition="$part" --job-name="ev_${ts}_${run}" --export=ALL \
                   "$CLONE_DIR/eval/run_suite.sbatch"
    done
done
echo
[ "$DRY" = 1 ] && echo "DRY=1: $n scoring arrays would be queued, nothing submitted" \
               || echo "$n scoring arrays queued (20 shards each). When squeue is empty: bash eval/finish_official_eval.sh"
