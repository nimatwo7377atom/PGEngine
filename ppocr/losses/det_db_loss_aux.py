# Place this file in: ppocr/losses/det_db_loss_aux.py
# Then register it in ppocr/losses/__init__.py:
#   from .det_db_loss_aux import DBLossWithAux
#
# VERIFY: adjust the import below ("from .det_db_loss import DBLoss") to
# match wherever DBLoss actually lives in your PaddleOCR checkout.

import paddle.nn as nn
import paddle.nn.functional as F
from .det_db_loss import DBLoss


class DBLossWithAux(nn.Layer):
    """Stock DBLoss + a weighted MSE loss on the auxiliary corner heatmap.

    aux_loss_weight: start around 0.3-0.5. If val hmean doesn't improve
    over plain DBLoss after a proper run, this whole idea isn't paying for
    itself on your data - just switch Loss/Head back to DBLoss/DBHead.
    """

    def __init__(self,
                 aux_loss_weight=0.5,
                 balance_loss=True,
                 main_loss_type='DiceLoss',
                 alpha=5,
                 beta=10,
                 ohem_ratio=3,
                 eps=1e-6,
                 **kwargs):
        super().__init__()
        self.db_loss = DBLoss(
            balance_loss=balance_loss,
            main_loss_type=main_loss_type,
            alpha=alpha,
            beta=beta,
            ohem_ratio=ohem_ratio,
            eps=eps)
        self.aux_loss_weight = aux_loss_weight

    def forward(self, predicts, batch):
        # ──────────────────────────────────────────────────────────
        # STEP 1: Extract corner_heatmap BEFORE passing to DBLoss.
        #         DBLoss only expects the standard 5 fields and will
        #         crash with "too many values to unpack" if it sees
        #         our extra field.
        # ──────────────────────────────────────────────────────────
        corner_target = None

        if isinstance(batch, dict):
            # Newer PaddleOCR: batch is a dict from KeepKeys
            corner_target = batch.pop('corner_heatmap', None)
        elif isinstance(batch, (list, tuple)):
            # Older PaddleOCR: batch is a list/tuple
            # Standard: [image, thresh_map, thresh_mask, shrink_map, shrink_mask]
            # Ours adds: corner_heatmap as the 6th element
            if len(batch) > 5:
                corner_target = batch[-1]
                batch = batch[:5]

        # ──────────────────────────────────────────────────────────
        # STEP 2: Compute standard DB loss (batch is now clean)
        # ──────────────────────────────────────────────────────────
        losses = self.db_loss(predicts, batch)

        # ──────────────────────────────────────────────────────────
        # STEP 3: Compute auxiliary corner heatmap loss
        # ──────────────────────────────────────────────────────────
        if (self.aux_loss_weight > 0
                and corner_target is not None
                and 'aux_corner_heatmap' in predicts):

            pred = predicts['aux_corner_heatmap']
            target = corner_target

            # Resize prediction to match target spatial dims if needed
            if tuple(pred.shape[2:]) != tuple(target.shape[2:]):
                pred = F.interpolate(
                    pred,
                    size=target.shape[2:],
                    mode='bilinear',
                    align_corners=False,
                )

            aux_loss = F.mse_loss(pred, target)
            losses['loss'] = losses['loss'] + self.aux_loss_weight * aux_loss
            losses['aux_loss'] = aux_loss

        return losses
