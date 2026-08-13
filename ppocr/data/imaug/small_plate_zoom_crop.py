import random
import cv2
import numpy as np


def _quad_area(q):
    x, y = q[:, 0], q[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


class SmallPlateZoomCrop(object):
    def __init__(self, p=0.5, target_short=64, image_shape=960, **kwargs):
        self.p = p
        self.target_short = target_short
        self.image_shape = int(image_shape)

    @staticmethod
    def _short_side(poly):
        p = np.asarray(poly, np.float32)
        q = np.roll(p, -1, axis=0)
        return float(min(np.linalg.norm(q - p, axis=1)))

    def __call__(self, data):
        if random.random() > self.p or len(data['polys']) == 0:
            return data
        img = data['image']
        H, W = img.shape[:2]
        polys = [np.asarray(q, np.float32) for q in data['polys']]
        shorts = [self._short_side(q) for q in polys]
        idx = random.choices(range(len(polys)),
                             weights=[1.0 / (s + 8.0) for s in shorts])[0]
        win = int(np.clip(shorts[idx] * self.image_shape / self.target_short,
                          96, max(H, W)))
        if win >= max(H, W):
            return data
        c = polys[idx].mean(axis=0)
        x0 = int(np.clip(c[0] - win / 2, 0, W - win))
        y0 = int(np.clip(c[1] - win / 2, 0, H - win))
        crop = img[y0:y0 + win, x0:x0 + win]

        # ROOT-CAUSE FIX: deliver a FULL-SIZE canvas to SASTProcessTrain.
        # Upscale the crop ourselves and scale polys by the same factor, so
        # image and labels stay perfectly registered and the sample looks
        # like an ordinary full-size image to the downstream crop logic.
        S = self.image_shape
        scale = S / float(win)
        crop = cv2.resize(crop, (S, S), interpolation=cv2.INTER_LINEAR)

        new_polys, new_tags = [], []
        for q, t in zip(polys, data['ignore_tags']):
            q = (q - [x0, y0]) * scale
            if (q[:, 0] >= 1).all() and (q[:, 0] <= S - 2).all() and \
               (q[:, 1] >= 1).all() and (q[:, 1] <= S - 2).all() and \
               _quad_area(q) > 16.0:
                new_polys.append(q)
                new_tags.append(t)
        if len(new_polys) == 0:
            return data
        data['image'] = crop
        data['polys'] = np.array(new_polys, np.float32)
        data['ignore_tags'] = new_tags
        return data
