#!/bin/bash
# LFCC-LCNN over every condition, then converted into the harness CSV schema.
#
#   bash eval/run_lcnn.sh                       # all ten conditions
#   CONDITIONS="RAW P0" bash eval/run_lcnn.sh   # a subset
#
# LCNN does not go through harness.py: it is a project-NN-Pytorch-scripts project
# whose model.py expects that framework's data plumbing and computes LFCC inside
# forward(). Wrapping it in the adapter shape would mean reimplementing that
# convention, which is the kind of guesswork that produces confident wrong
# numbers. Instead it is driven by its own tool and the output is converted --
# see eval/adapters/lcnn_NOTES.md.
#
# CPU-BOUND. LFCC extraction, not the GPU, sets the pace (~1 h per condition on a
# P100 per the baseline's own README). It therefore does not compete with the GPU
# queue and should be started early and left running.
#
# 8 kHz CONDITIONS. LCNN's config hardcodes wav_samp_rate = 16000 as a constant,
# not a conversion, so B1 and C1 must be fed from 16 kHz copies made by
# eval/upsample_nb.py -- with the same resampler harness.py uses, so LCNN sees
# the same samples as the other five detectors.

set -u

CLONE_DIR=${CLONE_DIR:-$HOME/Narrow-Frequency-Range-Clone}
MODELS=${MODELS:-$HOME/models}
LCNN_REPO=${LCNN_REPO:-$MODELS/asvspoof2021/LA/Baseline-LFCC-LCNN}
CKPT=${CKPT:-$LCNN_REPO/project/baseline_LA/__pretrained/trained_network.pt}

RAW_ROOT=${RAW_ROOT:-/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA}
NB_ROOT=${NB_ROOT:-/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019_NB}
NFR_ROOT=${NFR_ROOT:-/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019_NFR}
NB16_ROOT=${NB16_ROOT:-/scratch/hafiz_root/hafiz1/$USER/AsvSpoofData_2019_NB_16k}
LABELS=${LABELS:-$HOME/asvspoof_protocols/ASVspoof2019.LA.cm.eval.trl.txt}

CONDITIONS=${CONDITIONS:-"RAW P0 B1 C1 F0 F1 F2 F3 F4 F5"}
OUT_DIR=${OUT_DIR:-$CLONE_DIR/runs/suite}
SUBSET=${SUBSET:-eval}

[ -f "$CKPT" ] || { echo "checkpoint not found: $CKPT"; echo "run project/00_download.sh"; exit 1; }
[ -f "$LCNN_REPO/project/baseline_LA/02_eval_alternative.sh" ] || {
    echo "02_eval_alternative.sh not found under $LCNN_REPO"; exit 1; }

mkdir -p "$OUT_DIR/scores" "$OUT_DIR/lcnn_raw"

# where each condition's audio lives
audio_dir() {
    case "$1" in
      RAW)              echo "$RAW_ROOT/ASVspoof2019_LA_${SUBSET}/flac" ;;
      P0)               echo "$NB_ROOT/ASVspoof2019_LA_${SUBSET}/$1/flac" ;;
      B1|C1)            echo "$NB16_ROOT/ASVspoof2019_LA_${SUBSET}/$1/flac" ;;   # 16 kHz copies
      # LP<hz>/HP<hz>: the sub-band sweep from eval/build_subband.py, same tree as F*
      F0|F1|F2|F3|F4|F5|LP[0-9]*|HP[0-9]*) echo "$NFR_ROOT/ASVspoof2019_LA_${SUBSET}/$1/flac" ;;
      *) echo "" ;;
    esac
}

fail=0
for COND in $CONDITIONS; do
    DIR=$(audio_dir "$COND")
    [ -n "$DIR" ] || { echo "unknown condition '$COND'"; fail=1; continue; }
    if [ ! -d "$DIR" ]; then
        echo "--- $COND: $DIR missing, skipping ---"
        case "$COND" in B1|C1)
            echo "    run: python eval/upsample_nb.py --nb-root \$NB_ROOT --out $NB16_ROOT --cond $COND" ;;
        esac
        fail=1; continue
    fi

    # same opt-out harness.py honours, so one file drops a condition for all six models
    if [ -f "$OUT_DIR/scores/SKIP_CONDITIONS" ] && grep -qw "$COND" "$OUT_DIR/scores/SKIP_CONDITIONS"; then
        echo "--- $COND: listed in $OUT_DIR/scores/SKIP_CONDITIONS, skipping ---"
        continue
    fi

    OUT="$OUT_DIR/scores/lcnn_${COND}.csv"
    if [ -s "$OUT" ]; then
        echo "--- $COND: $OUT exists, skipping ---"
        continue
    fi

    # SET_PREFIX keeps corpora apart: the baseline names its score file (and any
    # cache) after the set, so two corpora sharing a condition name would collide.
    SET_NAME="${SET_PREFIX:-nfr}_${COND}"
    echo
    echo "=== lcnn / $COND ==="
    echo "    audio $DIR"

    ( cd "$LCNN_REPO/project/baseline_LA" \
      && bash 02_eval_alternative.sh "$DIR" "$SET_NAME" "$CKPT" ) || {
        echo "FAILED: lcnn / $COND (02_eval_alternative.sh)"; fail=1; continue; }

    # The baseline writes a score file named after the data set. Its exact name
    # varies between releases of project-NN, so take the newest match rather than
    # hardcoding one, and fail loudly if nothing appears.
    SCORES=$(ls -t "$LCNN_REPO/project/baseline_LA"/*"${SET_NAME}"* 2>/dev/null \
             | grep -v '\.sh$' | head -1)
    if [ -z "$SCORES" ]; then
        echo "FAILED: lcnn / $COND -- no score file matching '*${SET_NAME}*'"
        echo "    look in $LCNN_REPO/project/baseline_LA and pass it to lcnn_to_csv.py by hand"
        fail=1; continue
    fi
    cp "$SCORES" "$OUT_DIR/lcnn_raw/${SET_NAME}.txt"

    python "$CLONE_DIR/eval/lcnn_to_csv.py" \
        --scores "$SCORES" \
        --labels "$LABELS" \
        --cond "$COND" \
        --out "$OUT" || { echo "FAILED: lcnn_to_csv / $COND"; fail=1; }
done

echo
echo "score CSVs in $OUT_DIR/scores/ (lcnn_*.csv), raw LCNN output kept in $OUT_DIR/lcnn_raw/"
echo "metrics:  python eval/compute_metrics.py $OUT_DIR/scores/lcnn_*.csv --per-attack"
echo
echo "CHECK THE SIGN on the first condition: RAW should give a LOW EER. An EER"
echo "near 100-x% means higher scores mean more spoof -- rerun lcnn_to_csv.py"
echo "with --negate rather than adjusting anything downstream."
exit $fail
