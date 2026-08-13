# tools/label_audit.py
# فقط گزارش می‌دهد؛ هیچ فایلی را تغییر نمی‌دهد.
# اجرا:
#   python tools/label_audit.py --label my_dataset/train.txt
#   (اختیاری برای بررسی خارج‌بودن از تصویر:)
#   python tools/label_audit.py --label my_dataset/train.txt --data_dir /content/my_dataset
import argparse, json, os
from collections import Counter
import numpy as np

EPS = 1e-9

def orient(p, q, r):
    v = (q[0]-p[0])*(r[1]-p[1]) - (q[1]-p[1])*(r[0]-p[0])
    return -1 if v < -EPS else (1 if v > EPS else 0)

def seg_x(a, b, c, d):                      # برخورد دو یال غیرمجاور
    o1, o2 = orient(a, b, c), orient(a, b, d)
    o3, o4 = orient(c, d, a), orient(c, d, b)
    return o1 != o2 and o3 != o4

def signed_area(q):
    x, y = q[:, 0], q[:, 1]
    return 0.5*(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def convex(q):
    s = {orient(q[i], q[(i+1) % 4], q[(i+2) % 4]) for i in range(4)}
    s.discard(0)
    return len(s) <= 1

def metrics(q):
    e = [float(np.hypot(*(q[(i+1) % 4] - q[i]))) for i in range(4)]
    w, h = (e[0]+e[2])/2, (e[1]+e[3])/2
    a = abs(signed_area(q))
    short, long_ = min(w, h), max(w, h)
    return dict(w=w, h=h, area=a, short=short, long=long_,
                aspect=long_/max(short, 1e-6))

ap = argparse.ArgumentParser()
ap.add_argument('--label', required=True)
ap.add_argument('--data_dir', default='', help='اختیاری: بررسی محدوده‌ی تصویر')
ap.add_argument('--min_area', type=float, default=20)
ap.add_argument('--tiny', type=float, default=20)
args = ap.parse_args()

_sizes = {}
def img_size(name):
    if not args.data_dir:
        return None
    if name not in _sizes:
        import cv2
        im = cv2.imread(os.path.join(args.data_dir, name))
        _sizes[name] = None if im is None else (im.shape[1], im.shape[0])
    return _sizes[name]

recs, n_err, n_warn = [], 0, 0
for ln, line in enumerate(open(args.label, encoding='utf-8'), 1):
    line = line.strip()
    if not line:
        continue
    name, js = line.split('\t', 1)
    try:
        objs = json.loads(js)
    except Exception:
        print(f'line {ln:>4} [ERROR,BADJSON] {name}')
        continue
    for o in objs:
        q = np.asarray(o.get('points', []), float)
        issues, m, wind, vert = [], None, '?', False
        if q.ndim != 2 or q.shape != (4, 2):
            issues.append(('ERROR', 'NOT4', f'shape={q.shape}'))
        else:
            m = metrics(q)
            wind = 'CW' if signed_area(q) > 0 else 'CCW'   # CW = همان ترتیب رایج TL,TR,BR,BL
            vert = m['w'] < m['h']                          # قرارداد «پلاک عمودی»
            if seg_x(q[0], q[1], q[2], q[3]) or seg_x(q[1], q[2], q[3], q[0]):
                issues.append(('ERROR', 'BOWTIE', ''))      # خودمتقاطع = ترتیب نقاط غلط
            if m['area'] < args.min_area:
                issues.append(('ERROR', 'ZEROAREA', f"area={m['area']:.1f}"))
            if any(np.hypot(*(q[(i+1) % 4] - q[i])) < 1.0 for i in range(4)):
                issues.append(('ERROR', 'DUPPTS', ''))      # دو نقطه‌ی منطبق
            sz = img_size(name)
            if sz and ((q[:, 0] < -2).any() or (q[:, 0] > sz[0]+2).any() or
                       (q[:, 1] < -2).any() or (q[:, 1] > sz[1]+2).any()):
                issues.append(('ERROR', 'OUTIMG', f'img={sz[0]}x{sz[1]}'))
            if m['short'] < args.tiny:
                issues.append(('WARN', 'TINY', f"short={m['short']:.1f}"))
            if m['aspect'] > 9 or m['aspect'] < 1.2:
                issues.append(('WARN', 'ASPECT', f"aspect={m['aspect']:.2f}"))
            if not convex(q):
                issues.append(('WARN', 'CONCAVE', ''))
        recs.append(dict(ln=ln, name=name, pts=o.get('points'), m=m,
                         wind=wind, vert=vert, issues=issues))

# جهتِ اقلیت = مشکوک (برچسب‌ها باید یک‌دست باشند)
wc = Counter(r['wind'] for r in recs if r['m'])
maj = wc.most_common(1)[0][0] if wc else 'CW'
for r in recs:
    if r['m'] and r['wind'] != maj:
        r['issues'].append(('WARN', 'WINDMIN', f"wind={r['wind']} vs {maj}"))

for r in recs:
    if not r['issues']:
        continue
    if any(s == 'ERROR' for s, _, _ in r['issues']):
        n_err += 1
    else:
        n_warn += 1
    tags = ', '.join(t + (f'({d})' if d else '') for _, t, d in r['issues'])
    print(f"line {r['ln']:>4} [{tags}] {r['name']}")
    if r['m']:
        m = r['m']
        print(f"         pts={r['pts']}")
        print(f"         w={m['w']:.1f} h={m['h']:.1f} area={m['area']:.1f} "
              f"aspect={m['aspect']:.2f} wind={r['wind']} vert={'Y' if r['vert'] else 'N'}")

print('\n=== SUMMARY ===')
print(f'total objects={len(recs)}  lines_with_ERROR={n_err}  lines_with_WARN={n_warn}')
for t, c in Counter(t for r in recs for _, t, _ in r['issues']).most_common():
    print(f'  {t}: {c}')
print(f'winding: {dict(wc)}   vertical-order(w<h): {sum(r["vert"] for r in recs)}')
shorts = sorted(r['m']['short'] for r in recs if r['m'])
print('short-side histogram:')
for lo, hi in [(0, 10), (10, 20), (20, 30), (30, 45), (45, 70), (70, 1e9)]:
    print(f'  [{lo:>2},{hi if hi < 1e9 else "inf":>3}): '
          f'{sum(1 for s in shorts if lo <= s < hi)}')
print('10 smallest plates:')
for r in sorted((r for r in recs if r['m']), key=lambda r: r['m']['short'])[:10]:
    print(f"  line {r['ln']:>4} short={r['m']['short']:.1f}  {r['name']}")
