#!/bin/bash
# Pull every variant of one base Trump clip from Great Lakes down to web/audio/raw/.
#
#   GL_HOST=uniqname@greatlakes.arc-ts.umich.edu bash tools/fetch_clips.sh 00150
#
# One ssh round trip: find every wav whose name starts with Donald_Trump_<id>
# anywhere under DATA_ROOT, tar them up remotely, untar here. The attack folder
# is preserved, so you get web/audio/raw/{F5TTS,XTTSV2,...,-}/Donald_Trump_<id>*.wav
set -euo pipefail

BASE_ID=${1:?usage: GL_HOST=user@host bash tools/fetch_clips.sh <base-id>   e.g. 00150}
GL_HOST=${GL_HOST:?set GL_HOST, e.g. GL_HOST=uniqname@greatlakes.arc-ts.umich.edu}
REMOTE_ROOT=${REMOTE_ROOT:-/nfs/turbo/umd-hafiz/issf_server_data/famousfigures/Donald_Trump}
SPEAKER=${SPEAKER:-Donald_Trump}

DEST="$(cd "$(dirname "$0")/.." && pwd)/web/audio/raw"
mkdir -p "$DEST"

echo "host:   $GL_HOST"
echo "remote: $REMOTE_ROOT"
echo "clip:   ${SPEAKER}_${BASE_ID}*"
echo "into:   $DEST"
echo

ssh "$GL_HOST" "cd '$REMOTE_ROOT' && find . -type f -name '${SPEAKER}_${BASE_ID}*.wav' -print0 | tar -czf - --null -T -" \
    | tar -xzvf - -C "$DEST"

echo
echo "got:"
find "$DEST" -name "${SPEAKER}_${BASE_ID}*.wav" -printf '  %P\n' 2>/dev/null \
    || find "$DEST" -name "${SPEAKER}_${BASE_ID}*.wav"
echo
echo "next:  python tools/build_web_assets.py"
