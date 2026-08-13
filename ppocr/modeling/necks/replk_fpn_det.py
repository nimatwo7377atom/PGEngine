# Place this file in: ppocr/modeling/necks/replk_fpn_det.py
# Then register in ppocr/modeling/necks/__init__.py (see below).

from .db_fpn import RepLKFPN


class RepLKFPN_Det(RepLKFPN):
    """RepLKFPN that ALWAYS returns the fused tensor.

    PP-OCRv6's RepLKFPN returns {"fuse": ..., "aux_p2/p3/p4": ...} when
    self.training (for its deep-supervision loss). Run A does not wire those
    aux outputs, and SASTHead would crash on a dict - so we strip them here.
    (Run B may instead keep them for auxiliary supervision.)

    Inference cost is identical to RSEFPN-style necks after reparameterization;
    during training the multi-branch dilated convs learn a larger receptive
    field, which is the whole point of this run.
    """

    def forward(self, x):
        out = super().forward(x)
        if isinstance(out, dict):
            return out["fuse"]
        return out
