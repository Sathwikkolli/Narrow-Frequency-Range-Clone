#!/bin/bash
# Merge and score everything eval/submit_official_eval.sh produced, then print one
# EER table: rows = runs (model x training data), columns = test set x condition.
# Login node, no GPU:
#
#   bash eval/finish_official_eval.sh
#
# Per run and test set: merge the 20 shards per condition, check the trial count
# (71,237 for asv19, 31,778 for itw), then compute_metrics.py writes
#   results/official_eval/<testset>/<run>.{json,txt}
# A condition with missing shards or the wrong trial count is reported INCOMPLETE
# and left out of the JSON rather than scored on a partial trial list.

set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
CONDS=${CONDS:-"RAW P0 B1 C1"}
ROOT=runs/official_eval

[ -d "$ROOT" ] || { echo "no $ROOT -- run eval/submit_official_eval.sh first"; exit 1; }

for tsdir in "$ROOT"/*/; do
    ts=$(basename "$tsdir")
    case "$ts" in asv19) expected=71237 ;; itw) expected=31778 ;; *) continue ;; esac
    mkdir -p "results/official_eval/$ts"
    for rundir in "$tsdir"*/; do
        run=$(basename "$rundir")
        model=${run%%_*}
        [ -d "$rundir/scores" ] || continue
        EXPECTED=$expected bash eval/merge_suite.sh "$PWD/$rundir" > "$rundir/merge.txt" 2>&1
        files=() ; missing=""
        for c in $CONDS; do
            f="$rundir/${model}_${c}.csv"
            nsh=$(ls "$rundir"/scores/${model}_${c}_[0-9][0-9][0-9][0-9].csv 2>/dev/null | wc -l)
            rows=$([ -f "$f" ] && echo $(( $(wc -l < "$f") - 1 )) || echo 0)
            if [ "$nsh" -eq 20 ] && [ "$rows" -eq "$expected" ]; then
                files+=("$f")
            else
                missing="$missing $c(${nsh}/20 shards, $rows rows)"
            fi
        done
        if [ ${#files[@]} -gt 0 ]; then
            python eval/compute_metrics.py "${files[@]}" --per-attack \
                --json-out "results/official_eval/$ts/$run.json" \
                > "results/official_eval/$ts/$run.txt" 2>&1 \
                || echo "  $ts / $run: compute_metrics FAILED (see results/official_eval/$ts/$run.txt)"
        fi
        [ -n "$missing" ] && echo "  $ts / $run INCOMPLETE:$missing"
    done
done

echo
python - "$CONDS" <<'EOF'
import glob, json, os, sys
conds = sys.argv[1].split()
res = {}
for f in glob.glob("results/official_eval/*/*.json"):
    ts, run = f.split(os.sep)[-2], os.path.basename(f)[:-5]
    for r in json.load(open(f)):
        res[(run, ts, r["cond"])] = r["eer"] * 100
if not res:
    sys.exit("no results yet")
runs = sorted({k[0] for k in res}, key=lambda r: (r.split("_")[0], r))
tss = [t for t in ("asv19", "itw") if any(k[1] == t for k in res)]
head = "".join(f"{t + ' ' + c:>11s}" for t in tss for c in conds)
print("EER %   (wb = trained on wideband 2019 LA, itu = trained on C1 telephone audio)")
print(f"{'run':26s}{head}")
for r in runs:
    cells = "".join(f"{res[(r, t, c)]:11.2f}" if (r, t, c) in res else f"{'-':>11s}"
                    for t in tss for c in conds)
    print(f"{r:26s}{cells}")
EOF
