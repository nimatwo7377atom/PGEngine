# Place this file in: ppocr/utils/corner_refine.py
# Usage: from ppocr.utils.corner_refine import refine_quad, warp_plate

import cv2
import numpy as np


def refine_corner(channel, cx, cy, win=16, min_peak=0.25, temperature=0.1):
    """Refine ONE corner to sub-pixel accuracy via windowed soft-argmax.

    channel     : (H, W) float array in [0,1] - one heatmap channel
    cx, cy      : coarse corner (DB box) in the SAME pixel coords as channel
    win         : search window radius around the coarse corner (px)
    min_peak    : if heatmap peak < this, trust the coarse corner (fallback)
    temperature : smaller = sharper weighting around the peak

    returns     : (x, y, peak)
    """
    H, W = channel.shape
    cx_i, cy_i = int(round(cx)), int(round(cy))
    x0, x1 = max(0, cx_i - win), min(W, cx_i + win + 1)
    y0, y1 = max(0, cy_i - win), min(H, cy_i + win + 1)
    window = channel[y0:y1, x0:x1]

    if window.size == 0:
        return float(cx), float(cy), 0.0

    peak = float(window.max())
    if peak < min_peak:
        return float(cx), float(cy), peak

    # soft-argmax: وزن نمایی حول قله + حذف پس‌زمینه
    e = np.exp((window - peak) / max(temperature, 1e-6))
    e[window < 0.25 * peak] = 0.0
    s = e.sum()
    if s <= 1e-8:
        return float(cx), float(cy), peak

    ys, xs = np.mgrid[0:window.shape[0], 0:window.shape[1]]
    rx = x0 + float((xs * e).sum() / s)
    ry = y0 + float((ys * e).sum() / s)
    return rx, ry, peak


def refine_quad(quad, heatmap, win=16, min_peak=0.25, temperature=0.1,
                heatmap_resize_to=None):
    """Snap all 4 corners of a coarse DB quad onto the corner-heatmap peaks.

    quad    : (4, 2) array-like, order TL, TR, BR, BL (همان ترتیب لیبل‌ها)
    heatmap : (4, h, w) float in [0,1] - output of AuxCornerHead
    heatmap_resize_to : (H, W) optional - if heatmap is at a different
              resolution than your quad coords, it is resized first.

    returns : (refined_quad (4,2) float, peaks list of 4)
    """
    quad = np.asarray(quad, dtype=np.float64)
    assert quad.shape == (4, 2), "quad must be (4,2)"
    heatmap = np.asarray(heatmap, dtype=np.float64)
    assert heatmap.shape[0] == 4, "heatmap must have 4 channels"

    if heatmap_resize_to is not None:
        H, W = heatmap_resize_to
        if heatmap.shape[1] != H or heatmap.shape[2] != W:
            heatmap = np.stack([
                cv2.resize(heatmap[k], (W, H), interpolation=cv2.INTER_LINEAR)
                for k in range(4)
            ])

    out = quad.copy()
    peaks = []
    for k in range(4):
        rx, ry, p = refine_corner(
            heatmap[k], quad[k, 0], quad[k, 1],
            win=win, min_peak=min_peak, temperature=temperature)
        out[k, 0] = rx
        out[k, 1] = ry
        peaks.append(p)
    return out, peaks


def warp_plate(img, quad, out_w=520, out_h=110):
    """Perspective-rectify the plate using the refined quad.
    نسبت استاندارد پلاک ایران ~ 520x110.
    """
    src = np.asarray(quad, dtype=np.float32)
    dst = np.float32([[0, 0], [out_w, 0], [out_w, out_h], [0, out_h]])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (out_w, out_h))
