# Run once:
#   python tools/make_oversampled_train.py \
#       --src my_dataset/train.txt --dst my_dataset/train_oversampled.txt
import argparse
import json
import numpy as np


def short_side(pts):
    p = np.asarray(pts, float)
    q = np.roll(p, -1, axis=0)          # رأس بعدی؛ آخری به اولی برمی‌گردد
    return float(np.min(np.linalg.norm(q - p, axis=1)))


ap = argparse.ArgumentParser()
ap.add_argument('--src', default='my_dataset/train.txt')
ap.add_argument('--dst', default='my_dataset/train_oversampled.txt')
ap.add_argument('--thresh', type=float, default=30)   # short side px
ap.add_argument('--copies', type=int, default=3)
args = ap.parse_args()

n_small = 0
with open(args.src, encoding='utf-8') as f, \
        open(args.dst, 'w', encoding='utf-8') as g:
    for line in f:
        line = line.rstrip('\n')
        if not line.strip():
            continue
        name, js = line.split('\t', 1)
        if any(short_side(o['points']) < args.thresh for o in json.loads(js)):
            n_small += 1
            for _ in range(args.copies):
                g.write(line + '\n')
        g.write(line + '\n')
print(f'small-plate images: {n_small} -> each duplicated x{args.copies}')
