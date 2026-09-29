"""TCM-ADD -- Truong et al., Interspeech 2024 (NTU).

Source:   github.com/ductuantruong/tcm_add
Weights:  the repo's OneDrive -> interspeech24_tcm/LA/fixed_length/best_{0..4}.pth
          Five seeds, 1.20 GB each. LA has no avg_5_best.pth (only DF does).
Training: ASVspoof2019 LA train via `main.py --algo 5`, so 2019 LA eval is in-domain.
Published: EER 1.03% on ASVspoof 2021 LA, 2.06% on 2021 DF.

Taken from that repo, not guessed:

  window 66800      data_utils.py: `self.cut = 66800 # take ~4 sec audio`
                    NOTE this is NOT 64600 -- TCM wants ~2200 more samples than
                    AASIST and RawNet2. The harness takes the window from here.
  tile-repeat pad   utils.pad(): same tile-then-truncate rule
  score out[:, 1]   eval.py: `batch_score = (batch_out[:, 1])...ravel()`
  returns a TUPLE   model.py Model.forward returns (out, attn_score), so the
                    logits must be unpacked -- logits2() handles it
  arch args         eval.py defaults: emb-size 144, heads 4, kernel_size 31,
                    num_encoders 4. MyConformer's own default emb_size is 128, so
                    passing nothing would build the wrong shape. load_state_dict
                    is strict, so a mismatch fails loudly rather than silently.
  batching          eval runs at batch_size=10; real batches are fine

FP32 ONLY. model.py's SSLModel.extract_feat contains:

    if next(self.model.parameters()).device != input_data.device \\
       or next(self.model.parameters()).dtype != input_data.dtype:
        self.model.to(input_data.device, dtype=input_data.dtype)
        self.model.train()

The dtype branch fires under autocast/fp16/bf16 and silently puts XLS-R into
TRAIN mode -- dropout active -- during scoring. No error, just quietly wrong
numbers. Hence half_ok = False; the harness refuses fp16/bf16 for this model.
"""
from ._common import load_model, logits2, ns


class TcmAdapter:
    name = "tcm-add"
    window = 66800
    batch_ok = True
    half_ok = False           # see the docstring: bf16/fp16 flips XLS-R to train()
    gate_name = None

    def build(self, device, ckpt, repo=None):
        if not repo:
            import sys
            sys.exit("tcm needs --repo pointing at a clone of "
                     "github.com/ductuantruong/tcm_add")
        # eval.py's defaults -- these define the architecture, so they must match
        # the checkpoint exactly
        args = ns(emb_size=144, heads=4, kernel_size=31, num_encoders=4)
        model = load_model(repo, "model", "Model", args, device, ckpt)
        # Belt and braces against the train() branch above: even in fp32 the
        # device check can fire on the first forward.
        model.eval()
        if hasattr(model, "ssl_model") and hasattr(model.ssl_model, "model"):
            model.ssl_model.model.eval()
        return model

    def score(self, model, x):
        out = logits2(model(x), self.name)      # forward returns (logits, attn)
        logit_spoof, logit_bonafide = out[:, 0], out[:, 1]
        return logit_bonafide, logit_spoof, logit_bonafide
