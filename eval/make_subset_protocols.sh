#!/bin/bash
# Cut every protocol the experiments use to one fixed fraction. Login node, once:
#
#   bash eval/make_subset_protocols.sh            # 40% -> $HOME/asvspoof_protocols_p40/
#   PCT=25 bash eval/make_subset_protocols.sh     # 25% -> $HOME/asvspoof_protocols_p25/
#
# Writes, into $OUT (default $HOME/asvspoof_protocols_p$PCT):
#   ASVspoof2019.LA.cm.train.trn.txt   training           (train/submit.sh SUBSET=p40)
#   ASVspoof2019.LA.cm.dev.trl.txt     dev, best epoch    (train/submit.sh SUBSET=p40)
#   ASVspoof2019.LA.cm.eval.trl.txt    ASV19 scoring      (eval/submit_official_eval.sh SUBSET=p40)
#   ITW.cm.eval.trl.txt                In-the-Wild scoring (same)
# plus SHA256SUMS. Same seed every time, so re-running reproduces the same files.

set -eu
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PCT=${PCT:-40}
SEED=${SEED:-0}
T=/nfs/turbo/umd-hafiz/issf_server_data
SRC_TRAINDEV=${SRC_TRAINDEV:-$T/AsvSpoofData_2019/train/LA/ASVspoof2019_LA_cm_protocols}
SRC_EVAL=${SRC_EVAL:-$HOME/asvspoof_protocols/ASVspoof2019.LA.cm.eval.trl.txt}
SRC_ITW=${SRC_ITW:-$T/ITW_16k/protocol/ITW.cm.eval.trl.txt}
OUT=${OUT:-$HOME/asvspoof_protocols_p$PCT}
FRAC=$(python -c "print($PCT / 100)")

fail=0
for f in "$SRC_TRAINDEV/ASVspoof2019.LA.cm.train.trn.txt" "$SRC_TRAINDEV/ASVspoof2019.LA.cm.dev.trl.txt" \
         "$SRC_EVAL" "$SRC_ITW"; do
    [ -f "$f" ] || { echo "MISSING $f"; fail=1; }
done
[ "$fail" = 0 ] || exit 1

mkdir -p "$OUT"
cut1() { python eval/make_subset.py --in "$1" --out "$OUT/$2" --fraction "$FRAC" --seed "$SEED"; }
cut1 "$SRC_TRAINDEV/ASVspoof2019.LA.cm.train.trn.txt" ASVspoof2019.LA.cm.train.trn.txt
cut1 "$SRC_TRAINDEV/ASVspoof2019.LA.cm.dev.trl.txt"   ASVspoof2019.LA.cm.dev.trl.txt
cut1 "$SRC_EVAL"                                      ASVspoof2019.LA.cm.eval.trl.txt
cut1 "$SRC_ITW"                                       ITW.cm.eval.trl.txt
(cd "$OUT" && sha256sum ./*.txt > SHA256SUMS)
echo "-> $OUT"
