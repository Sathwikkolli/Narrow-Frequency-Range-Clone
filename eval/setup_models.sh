#!/bin/bash
# Clone the six detectors' official repos and fetch every checkpoint that can be
# fetched from a script. Run on a LOGIN node (compute nodes have no internet).
#
#   bash eval/setup_models.sh
#   MODELS=/scratch/hafiz_root/hafiz1/$USER/models bash eval/setup_models.sh
#
# Official sources only -- authors' own repos or the ASVspoof organisers. Four of
# the six checkpoints live on Google Drive / OneDrive, which hand HTML back to
# wget; those are listed at the end for a browser download plus scp.
#
# Disk: ~7 GB with one TCM seed, ~12 GB with all five.

set -u

MODELS=${MODELS:-$HOME/models}
mkdir -p "$MODELS"
cd "$MODELS"

say() { printf '\n=== %s ===\n' "$*"; }

# ---------------------------------------------------------------- repos
say "repos into $MODELS"

clone() {  # clone <dir> <url>
    if [ -d "$1/.git" ]; then
        echo "  $1 already cloned"
    else
        git clone --depth 1 "$2" "$1" || { echo "  FAILED to clone $2"; return 1; }
    fi
}

clone asvspoof2021     https://github.com/asvspoof-challenge/2021.git
clone aasist_repo      https://github.com/TakHemlata/SSL_Anti-spoofing.git
clone tcm_repo         https://github.com/ductuantruong/tcm_add.git
clone sls_repo         https://github.com/QiShanZhang/SLSforADD.git
clone nes2net_repo     https://github.com/Liu-Tianchi/Nes2Net_ASVspoof_ITW.git

# RawNet2 and LCNN are folders inside the challenge repo
RAWNET2_REPO="$MODELS/asvspoof2021/LA/Baseline-RawNet2"
LCNN_REPO="$MODELS/asvspoof2021/LA/Baseline-LFCC-LCNN"

# ---------------------------------------------------------------- XLS-R
# One copy, symlinked into each SSL repo. Those repos hardcode the RELATIVE path
# 'xlsr2_300m.pt', so the constructor only works from inside the repo directory.
# The fine-tuned checkpoint overwrites these weights, but the file must exist
# because the architecture is built from it.
say "fairseq XLS-R 300M (1.2 GB)"
if [ -f "$MODELS/xlsr2_300m.pt" ]; then
    echo "  already present"
else
    wget -q --show-progress -O "$MODELS/xlsr2_300m.pt" \
        https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr2_300m.pt \
        || echo "  FAILED -- fetch it manually"
fi

for r in aasist_repo tcm_repo sls_repo nes2net_repo; do
    if [ -d "$MODELS/$r" ] && [ -f "$MODELS/xlsr2_300m.pt" ]; then
        ln -sf "$MODELS/xlsr2_300m.pt" "$MODELS/$r/xlsr2_300m.pt"
        echo "  linked into $r"
    fi
done

# ---------------------------------------------------------------- RawNet2
say "RawNet2 pretrained (direct download)"
if ls "$MODELS"/rawnet2_ckpt/*.pth >/dev/null 2>&1; then
    echo "  already present"
else
    mkdir -p "$MODELS/rawnet2_ckpt"
    wget -q --show-progress -O "$MODELS/rawnet2_ckpt/pre_trained_DF_RawNet2.zip" \
        https://www.asvspoof.org/asvspoof2021/pre_trained_DF_RawNet2.zip \
        && unzip -o -q "$MODELS/rawnet2_ckpt/pre_trained_DF_RawNet2.zip" \
                 -d "$MODELS/rawnet2_ckpt" \
        && echo "  unzipped:" && find "$MODELS/rawnet2_ckpt" -name '*.pth' \
        || echo "  FAILED -- fetch the zip manually"
fi

# ---------------------------------------------------------------- LCNN
say "LFCC-LCNN pretrained (repo's own downloader)"
if [ -d "$LCNN_REPO/project" ]; then
    ( cd "$LCNN_REPO/project" && bash 00_download.sh ) \
        && echo "  done" || echo "  00_download.sh failed -- run it by hand"
else
    echo "  $LCNN_REPO/project not found"
fi

# ---------------------------------------------------------------- summary
say "what still needs a browser + scp"
cat <<'EOF'
These four are on Google Drive / OneDrive, which refuse wget. Download locally,
then scp into the paths below.

  W2V2-AASIST   best_SSL_model_LA.pth
                drive folder linked from github.com/TakHemlata/SSL_Anti-spoofing
                -> $MODELS/aasist_ckpt/

  TCM-ADD       best_0.pth .. best_4.pth   (LA/fixed_length, 1.20 GB each)
                https://entuedu-my.sharepoint.com/:f:/g/personal/truongdu001_e_ntu_edu_sg/El7AV62BKkdKhOYCyB3s2EkBLr-aVdj0doH0HNj9mTIsGA
                NOTE: use LA/fixed_length -- the LA/ root holds unrelated work
                (Qwen, VibeVoice) and LA/pretrained_model is not TCM either.
                -> $MODELS/tcm_ckpt/

  W2V2-SLS      drive folder linked from github.com/QiShanZhang/SLSforADD
                -> $MODELS/sls_ckpt/

  Nes2Net-X     drive links in the README table of
                github.com/Liu-Tianchi/Nes2Net_ASVspoof_ITW
                (use the ASVspoof/ITW repo, NOT Liu-Tianchi/Nes2Net, which is
                 the singing-voice release with a WavLM front end)
                -> $MODELS/nes2net_ckpt/

Example:
  scp best_0.pth ksathwik@greatlakes.arc-ts.umich.edu:~/models/tcm_ckpt/
EOF

say "layout"
mkdir -p "$MODELS"/{aasist_ckpt,tcm_ckpt,sls_ckpt,nes2net_ckpt}
du -sh "$MODELS" 2>/dev/null
ls -1 "$MODELS"
echo
echo "RawNet2 repo: $RAWNET2_REPO"
echo "LCNN repo:    $LCNN_REPO"
