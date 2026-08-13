# Place this file in: ppocr/data/imaug/make_corner_heatmap.py
# Then register it in ppocr/data/imaug/__init__.py:
#   from .make_corner_heatmap import MakeCornerHeatmap
#
# Position it in the Train.dataset.transforms list RIGHT NEXT TO
# MakeBorderMap / MakeShrinkMap (i.e. AFTER EastRandomCropData), so it reads
# the same crop-adjusted polygon coordinates they use.

import numpy as np


class MakeCornerHeatmap(object):
    """Builds a 4-channel Gaussian heatmap target used ONLY as an auxiliary
    training signal: channel k lights up at the true location of plate
    corner k (order must match your labeling convention -
    top-left, top-right, bottom-right, bottom-left - since that's the order
    the quad-labeling tool's auto-sort already produces).

    This target is consumed by DBHeadWithAux / DBLossWithAux and is never
    used at inference time.
    """

    def __init__(self, radius=4, downsample_stride=1, **kwargs):
        self.radius = radius
        # If you know your Neck's output stride (commonly 4 for DB-style
        # FPN necks), set this to build the heatmap already at that
        # resolution instead of full crop resolution. Leaving it at 1 also
        # works: DBLossWithAux resizes predictions to match at loss time.
        self.stride = downsample_stride

    def _draw_gaussian(self, heatmap, cx, cy, radius):
        h, w = heatmap.shape
        cx, cy = int(round(cx)), int(round(cy))
        if cx < 0 or cy < 0 or cx >= w or cy >= h:
            return heatmap
        y_grid, x_grid = np.ogrid[-radius:radius + 1, -radius:radius + 1]
        sigma = radius / 3.0 + 1e-6
        gaussian = np.exp(-(x_grid * x_grid + y_grid * y_grid) / (2 * sigma * sigma))
        x_min, x_max = max(0, cx - radius), min(w, cx + radius + 1)
        y_min, y_max = max(0, cy - radius), min(h, cy + radius + 1)
        gx_min, gx_max = x_min - (cx - radius), x_max - (cx - radius)
        gy_min, gy_max = y_min - (cy - radius), y_max - (cy - radius)
        if x_max > x_min and y_max > y_min:
            heatmap[y_min:y_max, x_min:x_max] = np.maximum(
                heatmap[y_min:y_max, x_min:x_max],
                gaussian[gy_min:gy_max, gx_min:gx_max])
        return heatmap

    def __call__(self, data):
        img = data['image']
        h, w = img.shape[:2]
        hh, ww = max(1, h // self.stride), max(1, w // self.stride)
        polys = data['polys']
        ignore_tags = data.get('ignore_tags', [False] * len(polys))

        heatmap = np.zeros((4, hh, ww), dtype=np.float32)
        for poly, ignore in zip(polys, ignore_tags):
            if ignore or len(poly) != 4:
                continue
            for k in range(4):
                cx = poly[k][0] / self.stride
                cy = poly[k][1] / self.stride
                self._draw_gaussian(heatmap[k], cx, cy, self.radius)

        data['corner_heatmap'] = heatmap
        return data
