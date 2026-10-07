#!/bin/bash
# Queue one training run. Login node, from anywhere:
#
#   SMOKE=1 bash train/submit.sh tcm itu 1     # ~1 h check: 2000/1000 clips, 2 epochs
#   bash train/submit.sh tcm itu 1             # the real run: C1 telephone audio
#   bash train/submit.sh tcm wb 1              # matched wideband baseline
#   DRY=1 bash train/submit.sh tcm itu 1       # check paths, print, submit nothing
#   AFTER=<jobid> bash train/submit.sh ...     # start only if that job (preflight) succeeds
#   ACCUM=2 bash train/submit.sh ...           # batch split in 2 if it does not fit memory
#
# Arguments: <model> <itu|wb> <seed index 1|2|3>. Seed index k trains with the
# repo's own default seed + k - 1, so s1 is exactly the published seed.
#
# Run name <model>_<data>_s<k>, e.g. tcm_itu_s1:
#   checkpoints/logs  $SCR/itu_train/runs/<run>/      (scratch)
#   slurm logs        $SCR/itu_train/slurm_logs/
#   final best.pth    $FINAL_ROOT/<run>/               (turbo, copied on finish)
#
# CHAIN copies of the 7-hour job are queued with --dependency=singleton; each
# resumes where the last stopped and the spare ones exit at once. 50 epochs is
# the cap; at a worst case of ~1 h/epoch that is ~50 h, so 10 slots of 7 h cover it -- raise it if a run
# writes no DONE.json by the time the chain ends, and just submit again: a
# re-submission resumes from last.pth.

set -eu
cd "$(dirname "${BASH_SOURCE[0]}")/.."
CLONE_DIR=$PWD

MODEL=${1:?usage: submit.sh <model> <itu|wb> <seed index>}
DATA=${2:?usage: submit.sh <model> <itu|wb> <seed index>}
K=${3:?usage: submit.sh <model> <itu|wb> <seed index>}
SMOKE=${SMOKE:-0}
AFTER=${AFTER:-}
ACCUM=${ACCUM:-1}
DRY=${DRY:-0}
CHAIN=${CHAIN:-10}

T=/nfs/turbo/umd-hafiz/issf_server_data
SCR=${SCR:-/scratch/hafiz_root/hafiz1/$USER}
MODELS=${MODELS:-$HOME/models}
PROTOCOLS=${PROTOCOLS:-$T/AsvSpoofData_2019/train/LA/ASVspoof2019_LA_cm_protocols}
FINAL_ROOT=${FINAL_ROOT:-$T/itu_ckpt}

case "$MODEL" in
  tcm) REPO=${REPO:-$MODELS/tcm_repo}; BASE_SEED=1234; ENV=${CONDA_ENV:-ssl_spoof} ;;
  nes2net) REPO=${REPO:-$MODELS/nes2net_repo}; BASE_SEED=12345; ENV=${CONDA_ENV:-ssl_spoof} ;;
  # sls trains in ssl_spoof, not its own eval env: that env's torch has no A40 kernels
  sls) REPO=${REPO:-$MODELS/sls_repo}; BASE_SEED=1234; ENV=${CONDA_ENV:-ssl_spoof} ;;
  aasist) REPO=${REPO:-$MODELS/aasist_repo}; BASE_SEED=1234; ENV=${CONDA_ENV:-ssl_spoof} ;;
  *)   echo "unknown model '$MODEL' (tcm, nes2net, sls, aasist)"; exit 1 ;;
esac
case "$DATA" in
  itu) DATA_ROOT=${DATA_ROOT:-$T/AsvSpoofData_2019_NB}; SUB=C1/flac ;;
  wb)  DATA_ROOT=${DATA_ROOT:-$T/AsvSpoofData_2019/train/LA}; SUB=flac ;;
  *)   echo "data must be itu or wb, not '$DATA'"; exit 1 ;;
esac
case "$K" in 1|2|3) ;; *) echo "seed index must be 1, 2 or 3"; exit 1 ;; esac
SEED=$((BASE_SEED + K - 1))

RUN=${MODEL}_${DATA}_s${K}
EXTRA="--accum $ACCUM"
FINAL_DIR=$FINAL_ROOT/$RUN
TIME=""
if [ "$SMOKE" = "1" ]; then
    RUN=${RUN}_smoke
    EXTRA="$EXTRA --limit-train 2000 --limit-dev 1000 --max-epochs 2 --keep-last"
    FINAL_DIR=""                 # a smoke test is never copied to turbo
    TIME="--time=02:00:00"
    CHAIN=1
fi
RUN_DIR=$SCR/itu_train/runs/$RUN
LOGS=$SCR/itu_train/slurm_logs

# --- check everything before anything is queued
fail=0
need() { [ -e "$1" ] || { echo "MISSING $2: $1"; fail=1; }; }
need "$REPO"                                     "repo"
need "$REPO/xlsr2_300m.pt"                       "XLS-R in the repo dir (ln -s \$MODELS/xlsr2_300m.pt)"
need "$PROTOCOLS/ASVspoof2019.LA.cm.train.trn.txt" "train protocol"
need "$PROTOCOLS/ASVspoof2019.LA.cm.dev.trl.txt"   "dev protocol"
need "$DATA_ROOT/ASVspoof2019_LA_train/$SUB"     "train audio"
need "$DATA_ROOT/ASVspoof2019_LA_dev/$SUB"       "dev audio"
[ "$fail" = 0 ] || exit 1
if [ -f "$RUN_DIR/DONE.json" ]; then
    echo "$RUN already finished:"; cat "$RUN_DIR/DONE.json"; exit 0
fi

echo "run        $RUN   (seed $SEED, env $ENV)"
echo "audio      $DATA_ROOT/ASVspoof2019_LA_{train,dev}/$SUB"
echo "run dir    $RUN_DIR"
echo "final dir  ${FINAL_DIR:-<none, smoke test>}"
DEP=singleton
[ -n "$AFTER" ] && DEP="singleton,afterok:$AFTER"
echo "jobs       $CHAIN chained x 7 h on spgpu, batch split $ACCUM, dependency $DEP"
[ -f "$RUN_DIR/last.pth" ] && echo "resuming   from $RUN_DIR/last.pth"
[ "$DRY" = "1" ] && { echo "DRY=1: nothing submitted"; exit 0; }

mkdir -p "$RUN_DIR" "$LOGS"
export MODEL DATA SEED RUN_DIR REPO DATA_ROOT PROTOCOLS SCR CLONE_DIR FINAL_DIR EXTRA
export CONDA_ENV=$ENV
for _ in $(seq 1 "$CHAIN"); do
    # shellcheck disable=SC2086  # TIME is empty or one flag
    sbatch --job-name="$RUN" --dependency="$DEP" $TIME \
        --output="$LOGS/${RUN}_%j.log" --export=ALL \
        "$CLONE_DIR/train/train.sbatch"
done
echo "follow:  tail -f $LOGS/${RUN}_*.log"
