# Place this file in: ppocr/modeling/heads/det_sast_head_aux.py
# Register in ppocr/modeling/heads/__init__.py (inside build_head, det branch):
#     from .det_sast_head_aux import SASTHeadWithAux
#     and add "SASTHeadWithAux" to support_dict

import paddle.nn as nn
from .det_sast_head import SASTHead


class _AuxScoreHead(nn.Layer):
    """Tiny 2-conv head: 1-channel text-center-line score map at the
    resolution of one intermediate neck level. Train-time only."""

    def __init__(self, in_channels, mid=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2D(in_channels, mid, 3, 1, 1, bias_attr=False),
            nn.BatchNorm2D(mid),
            nn.ReLU(),
            nn.Conv2D(mid, 1, 1),
        )

    def forward(self, x):
        return self.net(x)   # raw logits; loss applies BCE-with-logits


class SASTHeadWithAux(nn.Layer):
    """SASTHead + deep-supervision score heads on RepLKFPN's aux levels.

    Train : neck returns {"fuse":..., "aux_p2":..., "aux_p3":..., ...}
            -> SASTHead runs on "fuse", aux heads on the requested levels,
            aux logits are attached to the output dict for SASTLossWithAux.
    Eval  : neck returns the plain fused tensor -> behaves EXACTLY like
            SASTHead; aux heads never execute; export/serving unchanged.
    """

    def __init__(self, in_channels, aux_levels=("p2", "p3"), **kwargs):
        super().__init__()
        self.sast_head = SASTHead(in_channels, **kwargs)
        self.aux_levels = list(aux_levels)
        self.aux_heads = nn.LayerDict({
            lv: _AuxScoreHead(in_channels) for lv in self.aux_levels
        })

    def forward(self, x, targets=None):
        if isinstance(x, dict):
            feats = dict(x)
            fuse = feats.pop("fuse")
            out = self.sast_head(fuse, targets=targets)
            for lv in self.aux_levels:
                if "aux_" + lv in feats:
                    out["aux_score_" + lv] = self.aux_heads[lv](feats["aux_" + lv])
            return out
        return self.sast_head(x, targets=targets)
