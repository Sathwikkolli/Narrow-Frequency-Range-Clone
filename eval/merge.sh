#!/bin/bash
# Merge the per-shard score CSVs from the array job into one CSV per condition,
# then sanity-check the trial count. Run on a login node; no GPU needed.
#
#   bash eval/merge.sh [OUT_DIR]

set -eu
OUT_DIR=${1:-${OUT_DIR:-$HOME/Narrow-Frequency-Range-Clone/runs/dfarena_eval}}
CONDITIONS=${CONDITIONS:-"P0 B1 C1"}

cd "$OUT_DIR/scores"
for cond in $CONDITIONS; do
    shards=( ${cond}_[0-9]*.csv )
    [ -e "${shards[0]}" ] || { echo "$cond: no shard files, skipping"; continue; }
    out="../${cond}.csv"
    head -1 "${shards[0]}" > "$out"
    for f in "${shards[@]}"; do
        tail -n +2 "$f" >> "$out"
    done
    n=$(( $(wc -l < "$out") - 1 ))
    u=$(tail -n +2 "$out" | cut -d, -f1 | sort -u | wc -l)
    b=$(tail -n +2 "$out" | awk -F, '$3=="bonafide"' | wc -l)
    echo "$cond: ${#shards[@]} shards -> $out   rows $n   unique ids $u   bonafide $b"
    [ "$n" = "$u" ] || echo "  WARNING: duplicate utt_ids present"
    [ "$n" = "71237" ] || echo "  NOTE: expected 71237 trials for the LA eval protocol"
done
