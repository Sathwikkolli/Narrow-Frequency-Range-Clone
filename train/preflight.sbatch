#!/bin/bash
#SBATCH --job-name=preflight
#SBATCH --account=hafiz1
#SBATCH --partition=spgpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus=1
#SBATCH --mem=48g
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/hafiz_root/hafiz1/%u/itu_train/slurm_logs/preflight_%j.log

# --- everything that could break a training run mid-way, tested up front ---
#
#   mkdir -p /scratch/hafiz_root/hafiz1/$USER/itu_train/slurm_logs
#   MODEL=tcm sbatch train/preflight.sbatch
#   MODEL=tcm ACCUM=2 sbatch train/preflight.sbatch      # if memory FAILs
#   MODEL=tcm SAMPLE=0 sbatch --time=3:00:00 train/preflight.sbatch   # decode ALL ~100k files
#
# Same partition, CPUs, memory and conda env as train.sbatch, so what passes here
# is what the real job gets. Four stages, each must pass:
#
#   1. train/preflight.py   env, resampler, every ITU (C1) and wideband train/dev
#                           file present + 300 per split decoded, GPU memory at the published
#                           batch, speed -> run-time projection, checkpoint
#                           save/reload, scratch and turbo space
#   2. stop + resume        the real train.py on a small subset: stopped after
#                           epoch 1 the way a time limit stops it, launched again,
#                           must resume and finish epoch 2 with DONE.json
#   3. eval compatibility   eval/harness.py scores 64 C1 eval clips with the
#                           best.pth that stage 2 wrote
#   4. home untouched       nothing from this job landed in $HOME
#
# About 5 min, most of it loading XLS-R four times.

set -u
MODEL=${MODEL:-tcm}
ACCUM=${ACCUM:-1}
SAMPLE=${SAMPLE:-300}
CLONE_DIR=${CLONE_DIR:-${SLURM_SUBMIT_DIR:-$HOME/Narrow-Frequency-Range-Clone}}
T=/nfs/turbo/umd-hafiz/issf_server_data
SCR=${SCR:-/scratch/hafiz_root/hafiz1/$USER}
MODELS=${MODELS:-$HOME/models}
PROTOCOLS=${PROTOCOLS:-$T/AsvSpoofData_2019/train/LA/ASVspoof2019_LA_cm_protocols}
ITU_ROOT=$T/AsvSpoofData_2019_NB
WB_ROOT=$T/AsvSpoofData_2019/train/LA
FINAL_ROOT=${FINAL_ROOT:-$T/itu_ckpt}
WORK=$SCR/itu_train/preflight/$MODEL

case "$MODEL" in
  tcm) REPO=${REPO:-$MODELS/tcm_repo}; ENV=${CONDA_ENV:-ssl_spoof} ;;
  nes2net) REPO=${REPO:-$MODELS/nes2net_repo}; ENV=${CONDA_ENV:-ssl_spoof} ;;
  sls) REPO=${REPO:-$MODELS/sls_repo}; ENV=${CONDA_ENV:-$SCR/envs/sls_a40} ;;
  aasist) REPO=${REPO:-$MODELS/aasist_repo}; ENV=${CONDA_ENV:-ssl_spoof} ;;
  rawnet2) REPO=${REPO:-$MODELS/asvspoof2021/LA/Baseline-RawNet2}; ENV=${CONDA_ENV:-wmcompare} ;;
  lcnn) REPO=${REPO:-$MODELS/asvspoof2021/LA/Baseline-LFCC-LCNN}; ENV=${CONDA_ENV:-ssl_spoof} ;;
  *)   echo "unknown model '$MODEL'"; exit 1 ;;
esac

export TMPDIR=$SCR/itu_train/cache/tmp XDG_CACHE_HOME=$SCR/itu_train/cache \
       TORCH_HOME=$SCR/itu_train/cache/torch HF_HOME=$SCR/itu_train/cache/hf \
       PIP_CACHE_DIR=$SCR/itu_train/cache/pip PYTHONUNBUFFERED=1
mkdir -p "$TMPDIR"
rm -rf "$WORK"; mkdir -p "$WORK"
touch "$WORK/.start"

eval "$(conda shell.bash hook)"
conda activate "$ENV"
echo "preflight  $MODEL  env $ENV  job ${SLURM_JOB_ID:-}  $(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

declare -A STAGE
W=${SLURM_CPUS_PER_TASK:-8}

echo; echo "################ 1. checks"
python "$CLONE_DIR/train/preflight.py" --model "$MODEL" \
    --data itu wb --data-root "$ITU_ROOT" "$WB_ROOT" \
    --repo "$REPO" --protocols "$PROTOCOLS" \
    --scratch "$SCR/itu_train" --final-root "$FINAL_ROOT" \
    --workers "$W" --accum "$ACCUM" --sample "$SAMPLE" \
    && STAGE[1]=PASS || STAGE[1]=FAIL

echo; echo "################ 2. stop + resume with the real trainer"
RUN=$WORK/run
ARGS=(--model "$MODEL" --data itu --seed 1 --run-dir "$RUN" --repo "$REPO"
      --data-root "$ITU_ROOT" --protocols "$PROTOCOLS" --workers "$W" --accum "$ACCUM"
      --limit-train 60 --limit-dev 40 --max-epochs 2)
python "$CLONE_DIR/train/train.py" "${ARGS[@]}" --stop-after-epoch 1
r1=$?
python "$CLONE_DIR/train/train.py" "${ARGS[@]}"
r2=$?
if [ $r1 -eq 0 ] && [ $r2 -eq 0 ] && [ -f "$RUN/DONE.json" ] && [ ! -f "$RUN/last.pth" ] \
   && [ "$(tail -n +2 "$RUN/train_log.csv" | wc -l)" -eq 2 ] \
   && [ "$(wc -l < "$RUN/starts.log")" -eq 2 ] && grep -q "from epoch 1" "$RUN/starts.log"; then
    STAGE[2]=PASS
else
    STAGE[2]=FAIL
    echo "exit codes $r1/$r2; run dir:"; ls -la "$RUN"; cat "$RUN/starts.log" "$RUN/train_log.csv" 2>/dev/null
fi

echo; echo "################ 3. eval harness loads the trained best.pth"
if [ -f "$RUN/best.pth" ]; then
    python "$CLONE_DIR/eval/harness.py" --adapter "$MODEL" --repo "$REPO" --ckpt "$RUN/best.pth" \
        --labels "$PROTOCOLS/ASVspoof2019.LA.cm.eval.trl.txt" \
        --manifest "$ITU_ROOT/manifest/C1_eval_"*.csv --nb-root "$ITU_ROOT" \
        --out "$WORK/eval_scores.csv" --limit 64 --batch-size 16 --workers "$W"
    n=$( [ -f "$WORK/eval_scores.csv" ] && tail -n +2 "$WORK/eval_scores.csv" | wc -l || echo 0)
    [ "$n" -ge 64 ] && STAGE[3]=PASS || { STAGE[3]=FAIL; echo "scored $n of 64"; }
else
    STAGE[3]=FAIL; echo "no best.pth from stage 2"
fi

echo; echo "################ 4. nothing written to home"
new=$(find "$HOME" -xdev -newer "$WORK/.start" -type f \
        -not -path "$CLONE_DIR/*" -not -path "$HOME/.conda/*" 2>/dev/null | head -20)
if [ -z "$new" ]; then STAGE[4]=PASS; else STAGE[4]=WARN; echo "new files in home:"; echo "$new"; fi
home-quota 2>/dev/null | tail -1

echo; echo "################ summary"
fail=0
for s in 1 2 3 4; do
    printf "  stage %s  %s\n" "$s" "${STAGE[$s]}"
    [ "${STAGE[$s]}" = FAIL ] && fail=1
done
if [ $fail -eq 0 ]; then
    echo "PREFLIGHT PASSED -- safe to: bash train/submit.sh $MODEL itu 1 && bash train/submit.sh $MODEL wb 1"
    rm -rf "$WORK"
else
    echo "PREFLIGHT FAILED -- fix the FAIL lines above; work dir kept: $WORK"
fi
exit $fail
