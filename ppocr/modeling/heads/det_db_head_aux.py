# Place this file in: ppocr/modeling/heads/det_db_head_aux.py
# (REPLACES the previous version - already registered in __init__.py)

import paddle
import paddle.nn as nn
from .det_db_head import DBHead


class AuxCornerHead(nn.Layer):
    """Auxiliary 4-channel corner heatmap branch.

    قطعه ۳: با aux_upsample=4 دو مرحله upsample 2x اضافه می‌شود تا خروجی
    به رزولوشن ورودی (stride 1) برسد و peak ها به‌جای ۴ پیکسل، روی ۱ پیکسل
    نشسته و تیز شوند. هزینه‌اش چند conv کوچک روی کانال‌های کم است.
    """

    def __init__(self, in_channels, upsample_factor=1, **kwargs):
        super().__init__()
        mid = max(8, in_channels // 4)
        layers = [
            nn.Conv2D(in_channels, mid, 3, padding=1),
            nn.BatchNorm2D(mid),
            nn.ReLU(),
        ]
        f = int(upsample_factor)
        while f > 1:  # 4 -> دو مرحله‌ی 2x
            layers += [
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2D(mid, mid, 3, padding=1),
                nn.BatchNorm2D(mid),
                nn.ReLU(),
            ]
            f //= 2
        layers.append(nn.Conv2D(mid, 4, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return paddle.nn.functional.sigmoid(self.net(x))


class DBHeadWithAux(nn.Layer):
    """قطعه ۱: با always_aux=True هد کمکی در استنتاج هم روشن می‌ماند.

    نکته: کلید 'maps' همچنان اول دیکشنری است، پس DBPostProcess و اسکریپت‌های
    فعلی شما بدون هیچ تغییری کار می‌کنند؛ کلید اضافه فقط برای مصرف‌کننده‌ی
    جدید (corner refinement) است.
    """

    def __init__(self, in_channels, k=50, always_aux=False, aux_upsample=1, **kwargs):
        super().__init__()
        self.db_head = DBHead(in_channels, k=k, **kwargs)
        self.aux_head = AuxCornerHead(in_channels, upsample_factor=aux_upsample)
        self.always_aux = bool(always_aux)

    def forward(self, x, targets=None):
        db_out = self.db_head(x, targets=targets)
        if self.training or self.always_aux:
            db_out["aux_corner_heatmap"] = self.aux_head(x)
        return db_out
