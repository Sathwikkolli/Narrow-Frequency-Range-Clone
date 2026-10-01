#!/bin/bash
# Morning after eval/submit_subband.sh: merge the score shards, check every model
# and condition is complete, and write the metric summaries. Login node, CPU only.
#
#   bash eval/finish_subband.sh
#
# Writes results/subband/<model>.{json,txt} for ASVspoof2019 and
# results/subband/itw_<model>.{json,txt} for In-the-Wild. A model with any
# condition missing or short is reported and gets no summary, so a partial run
# cannot pass for a complete one. Exit code is 0 only if everything is complete.

set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."

MODELS_ALL=${MODELS_ALL:-"rawnet2 lcnn aasist nes2net tcm sls"}
# One low-pass and the four high-pass cutoffs. LP3400 + HP300 are the two halves of
# F3. LP2000/4000/5000/6000/7000 were built too and can be passed back in via CONDS.
CONDS=${CONDS:-"LP3400 HP100 HP300 HP500 HP1000"}

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-wmcompare}"
mkdir -p results/subband

incomplete=0
finish() {  # finish <runs dir> <expected trials> <results prefix>
    local dir=$1 expected=$2 prefix=$3
    [ -d "$dir/scores" ] || { echo "== $dir: no scores folder, skipping"; incomplete=1; return; }
    echo "== $dir (expect $expected trials per condition)"

    EXPECTED=$expected bash eval/merge_suite.sh "$dir" > "$dir/merge.log" 2>&1 \
        || echo "   merge_suite.sh reported a problem, see $dir/merge.log"
    # LCNN is scored whole, not in shards, so merge_suite.sh does not see it
    for f in "$dir"/scores/lcnn_*.csv; do [ -e "$f" ] && cp "$f" "$dir/"; done

    for m in $MODELS_ALL; do
        local ok=1 note=""
        for c in $CONDS; do
            local f="$dir/${m}_${c}.csv" n=0
            [ -s "$f" ] && n=$(( $(wc -l < "$f") - 1 ))
            [ "$n" = "$expected" ] || { ok=0; note="$note $c=$n"; }
        done
        if [ "$ok" = 1 ]; then
            # shellcheck disable=SC2046
            python eval/compute_metrics.py $(for c in $CONDS; do echo "$dir/${m}_${c}.csv"; done) \
                --per-attack --json-out "results/subband/${prefix}${m}.json" \
                > "results/subband/${prefix}${m}.txt" \
                && echo "   $m  complete -> results/subband/${prefix}${m}.{json,txt}" \
                || { echo "   $m  compute_metrics FAILED"; incomplete=1; }
        else
            echo "   $m  INCOMPLETE:$note"
            incomplete=1
        fi
    done
}

finish runs/subband     71237 ""
finish runs/subband_itw 31778 "itw_"

echo
if [ "$incomplete" = 0 ]; then
    echo "all complete. Next:"
    echo "  git add results/subband && git commit -m 'Sub-band sweep results' && git push origin main"
else
    echo "some runs are incomplete (see above). Failed tasks:"
    echo "  sacct -S $(date -d yesterday +%F 2>/dev/null || echo now-1days) -X --state=FAILED,TIMEOUT,CANCELLED,OUT_OF_MEMORY --format=JobID%20,JobName,State,Elapsed"
    echo "re-run bash eval/submit_subband.sh to requeue only what is missing, then this script again."
fi
exit $incomplete
