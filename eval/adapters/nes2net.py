"""Nes2Net-X -- Liu et al., IEEE T-IFS.

Source:   github.com/Liu-Tianchi/Nes2Net_ASVspoof_ITW      <- the SPEECH repo
          NOT github.com/Liu-Tianchi/Nes2Net, which is the singing-voice
          (CtrSVDD) release with a WavLM front end. Its checkpoints are trained
          on singing through a different front end and would run here without
          erroring while producing meaningless scores.
Weights:  Google Drive links in that repo's README (the HuggingFace mirror
          nielsr/nes2net-checkpoints is a third-party upload, so prefer these).
Published: EER 1.73% on ASVspoof 2021 LA, 1.65% on 2021 DF, 5.52% In-the-Wild.

Taken from easy_inference_demo.py in that repo, not guessed:

  import          from model_scripts.wav2vec2_Nes2Net_X import \\
                      wav2vec2_Nes2Net_no_Res_w_allT as Model
  construction    Model(args, device), with the demo's own defaults
  score           `pred = model(x)[:, 1]`
  window 64000    `if args.test_mode == '4s': audio = pad(audio, 64000)`

  ^ NOTE 64000, not 64600. Nes2Net is the only model in the suite using this
    length, and the harness takes the window from here so the model sees what it
    was evaluated with.

  CAVEAT: the demo's DEFAULT test_mode is 'full', i.e. variable-length input, and
  the published numbers may come from that rather than the 4 s crop. A fixed
  window is required here to batch and to hold clip length constant across all
  six detectors, so '4s' is used. If Nes2Net's P0/RAW number comes out well above
  its published figure, this is the first thing to check.

Back-end is 511k parameters -- the lightest SSL back-end in the suite.
"""
from _common import load_model, logits2, ns


class Nes2NetAdapter:
    name = "nes2net-x"
    window = 64000            # the repo's '4s' mode; see the caveat above
    batch_ok = True
    half_ok = True
    gate_name = None

    def build(self, device, ckpt, repo=None):
        if not repo:
            import sys
            sys.exit("nes2net needs --repo pointing at a clone of "
                     "github.com/Liu-Tianchi/Nes2Net_ASVspoof_ITW "
                     "(the speech repo, NOT Liu-Tianchi/Nes2Net)")
        # easy_inference_demo.py's defaults; these define the architecture, so
        # they must match the checkpoint
        args = ns(n_output_logits=2, dilation=2, pool_func="mean",
                  SE_ratio=[1], Nes_ratio=[8, 8], AASIST_scale=32)
        return load_model(repo, "model_scripts.wav2vec2_Nes2Net_X",
                          "wav2vec2_Nes2Net_no_Res_w_allT", args, device, ckpt)

    def score(self, model, x):
        out = logits2(model(x), self.name)
        logit_spoof, logit_bonafide = out[:, 0], out[:, 1]
        return logit_bonafide, logit_spoof, logit_bonafide
