"""W2V2-SLS -- Zhang, Wen & Hu, ACM MM 2024.

Source:   github.com/QiShanZhang/SLSforADD
Weights:  the repo's Google Drive folder (Baidu mirror also given)
Training: ASVspoof2019 LA train, so 2019 LA eval is in-domain.
Published: EER 2.87% on ASVspoof 2021 LA, 1.92% on 2021 DF, 7.46% In-the-Wild.

Taken from that repo, not guessed:

  window 64600      uses the same data_utils_SSL.py lineage as SSL_Anti-spoofing,
                    whose pad() default and dataset `cut` are both 64600
  score out[:, 1]   main.py produce_evaluation_file():
                    `batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()`
  plain [B, 2]      batch_out feeds CrossEntropyLoss directly
  batching          eval runs at batch_size=8; real batches are fine

Environment differs from AASIST/TCM: same fairseq commit (a54021305d6b3c) but
torch 1.12.1 rather than 1.8.1, so it needs its own conda environment.

SLS is the heaviest back-end in the suite at 23,399k parameters -- roughly 50x
Nes2Net and 10x TCM. Its mechanism is selecting which XLS-R layers to attend to,
which makes it the most informative model here for the narrowband question: if
band-limiting shifts the evidence to different layers, this is the model that can
show it. Worth capturing gates if a gating module turns out to be tappable by
name; leaving gate_name None until the module tree is inspected.
"""
from ._common import load_model, logits2, ns


class SlsAdapter:
    name = "w2v2-sls"
    window = 64600
    batch_ok = True
    half_ok = False           # SSLModel.extract_feat calls self.model.train() on a
                              # dtype change, silently enabling dropout while scoring
    gate_name = None          # set once the module tree is inspected on the cluster

    def build(self, device, ckpt, repo=None):
        if not repo:
            import sys
            sys.exit("sls needs --repo pointing at a clone of "
                     "github.com/QiShanZhang/SLSforADD")
        args = ns(algo=3, seed=1234, batch_size=5, loss="WCE")
        return load_model(repo, "model", "Model", args, device, ckpt)

    def score(self, model, x):
        out = logits2(model(x), self.name)
        logit_spoof, logit_bonafide = out[:, 0], out[:, 1]
        return logit_bonafide, logit_spoof, logit_bonafide
