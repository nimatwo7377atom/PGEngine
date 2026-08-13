# Place this file in: ppocr/losses/det_sast_loss_aux.py
# Register in ppocr/losses/__init__.py (inside build_loss):
#     from .det_sast_loss_aux import SASTLossWithAux
#     and add "SASTLossWithAux" to support_dict

import paddle.nn as nn
import paddle.nn.functional as F
from .det_sast_loss import SASTLoss


class SASTLossWithAux(nn.Layer):
    """Stock SASTLoss + BCE deep-supervision on the aux score heads.

    Free at inference (aux heads don't run when training==False).
    aux_weight: start at 0.3-0.5; if Run B doesn't beat Run A on the
    small buckets, this idea isn't paying - fall back to Run A config.
    """

    def __init__(self, aux_weight=0.5, **kwargs):
        super().__init__()
        self.sast_loss = SASTLoss(**kwargs)
        self.aux_weight = aux_weight

    def forward(self, predicts, labels):
        aux_preds = {k: predicts.pop(k)
                     for k in list(predicts.keys())
                     if k.startswith("aux_score_")}
        losses = self.sast_loss(predicts, labels)

        if self.aux_weight > 0 and aux_preds:
            if isinstance(labels, dict):
                gt = labels["score_map"]
            else:                                   # older list-based batches
                gt = labels[1]                      # [image, score_map, ...]
            if gt.ndim == 3:
                gt = gt.unsqueeze(1)                # (B, 1, H, W)
            for name, pred in aux_preds.items():
                gt_r = F.interpolate(gt, size=pred.shape[2:],
                                     mode="bilinear", align_corners=False)
                l = F.binary_cross_entropy_with_logits(pred, gt_r)
                losses[name] = self.aux_weight * l
                losses["loss"] = losses["loss"] + self.aux_weight * l
        return losses
