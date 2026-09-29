"""W2V2-AASIST -- Tak et al., Odyssey 2022 (EURECOM).

Source:   github.com/TakHemlata/SSL_Anti-spoofing
Weights:  the repo's Google Drive folder -> best_SSL_model_LA.pth
Training: ASVspoof2019 LA train, so 2019 LA eval is in-domain.
Published: EER 0.82% on ASVspoof 2021 LA, 2.85% on 2021 DF.

Taken from that repo, not guessed:

  window 64600      data_utils_SSL.py: `self.cut = 64600 # take ~4 sec audio`
  tile-repeat pad   data_utils_SSL.py pad(): np.tile then truncate
  score out[:, 1]   main_SSL_LA.py produce_evaluation_file():
                    `batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()`
  plain [B, 2]      batch_out feeds CrossEntropyLoss directly, so it is a tensor
  batching          eval runs at batch_size=10; real batches are fine

Back-end is 447k parameters on top of a 315M XLS-R front end, so the front end
dominates cost. Needs fairseq at commit a54021305d6b3c and xlsr2_300m.pt in the
repo directory -- see _common.py.

This is the validation anchor for the whole suite: it is the best-documented
model here, so if the harness reproduces its published number the loader is
trustworthy for the rest.
"""
from ._common import load_model, logits2, ns


class AasistAdapter:
    name = "w2v2-aasist"
    window = 64600
    batch_ok = True
    half_ok = False           # SSLModel.extract_feat calls self.model.train() on a
                              # dtype change, silently enabling dropout while scoring
    gate_name = None          # no per-layer gating module in this back-end

    def build(self, device, ckpt, repo=None):
        if not repo:
            import sys
            sys.exit("aasist needs --repo pointing at a clone of "
                     "github.com/TakHemlata/SSL_Anti-spoofing")
        # Model(args, device) accepts the training Namespace; nothing in the
        # forward path reads these at inference, but they are passed for parity
        # with the repo's own construction.
        args = ns(algo=5, seed=1234, batch_size=14, loss="WCE")
        return load_model(repo, "model", "Model", args, device, ckpt)

    def score(self, model, x):
        out = logits2(model(x), self.name)
        logit_spoof, logit_bonafide = out[:, 0], out[:, 1]
        return logit_bonafide, logit_spoof, logit_bonafide
