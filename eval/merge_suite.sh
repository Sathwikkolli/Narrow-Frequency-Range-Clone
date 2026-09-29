#!/bin/bash
# Merge the per-shard score CSVs from run_suite.sbatch into one CSV per
# (adapter, condition), then sanity-check the trial count. Login node, no GPU.
#
#   bash eval/merge_suite.sh [OUT_DIR]
#
# Result:  $OUT_DIR/<adapter>_<cond>.csv   ready for compute_metrics.py

set -eu
OUT_DIR=${1:-${OUT_DIR:-$HOME/Narrow-Frequency-Range-Clone/runs/suite}}
EXPECTED=${EXPECTED:-71237}      # trials in the ASVspoof2019 LA eval protocol

cd "$OUT_DIR/scores"

# every <adapter>_<cond> pair that has at least one shard
pairs=$(ls -1 *_[0-9][0-9][0-9][0-9].csv 2>/dev/null | sed -E 's/_[0-9]{4}\.csv$//' | sort -u)
[ -n "$pairs" ] || { echo "no shard files in $OUT_DIR/scores"; exit 1; }

for p in $pairs; do
    shards=( ${p}_[0-9][0-9][0-9][0-9].csv )
    [ -e "${shards[0]}" ] || continue
    out="../${p}.csv"
    head -1 "${shards[0]}" > "$out"
    for f in "${shards[@]}"; do
        tail -n +2 "$f" >> "$out"
    done
    n=$(( $(wc -l < "$out") - 1 ))
    u=$(tail -n +2 "$out" | cut -d, -f1 | sort -u | wc -l)
    b=$(tail -n +2 "$out" | awk -F, '$3=="bonafide"' | wc -l)
    printf '%-24s %2d shards -> %-28s rows %6d  unique %6d  bonafide %5d\n' \
        "$p" "${#shards[@]}" "$(basename "$out")" "$n" "$u" "$b"
    [ "$n" = "$u" ] || echo "    WARNING: duplicate utt_ids"
    [ "$n" = "$EXPECTED" ] || echo "    NOTE: expected $EXPECTED trials, got $n"
done
