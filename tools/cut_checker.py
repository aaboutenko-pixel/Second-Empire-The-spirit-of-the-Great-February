# Cuts a drawing off a backdrop that is pure white or a drawn-in checkerboard (two light greys), keeping every light
# part enclosed by the drawing. Only the backdrop region connected to the picture border goes; along its edge the
# drawing's own colour is carried outward and the alpha is solved against the local backdrop colour (the square the
# pixel sits in). Usage: cut_checker.py <src> <out.png> <preview_dir>
import os
import sys

import cv2
import numpy as np
from PIL import Image

TOL = 4       # a pixel this close to a backdrop grey, with no tint, is backdrop for sure
NEAR = 26     # this close (and untinted) it may be backdrop or edge, when connected to the border

src, out, prev = sys.argv[1], sys.argv[2], sys.argv[3]
im = Image.open(src).convert('RGB')
c = np.array(im).astype(np.float32)
grey = c.mean(axis=2)
tint = c.max(axis=2) - c.min(axis=2)
# the backdrop greys: the two most common untinted light values along the border
border = np.concatenate([c[0], c[-1], c[:, 0], c[:, -1]])
bg = border[(border.max(axis=1) - border.min(axis=1)) <= 3].mean(axis=1).round().astype(int)
levels = [v for v, _ in sorted(((v, n) for v, n in zip(*np.unique(bg, return_counts=True))), key=lambda t: -t[1])[:2]]
print('backdrop greys:', levels)
d = np.min([np.abs(grey - v) for v in levels], axis=0)
pure = (d <= TOL) & (tint <= 3)
near = (d <= NEAR) & (tint <= 12)
n, lab = cv2.connectedComponents(near.astype(np.uint8), connectivity=4)
edge_ids = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
back = np.isin(lab, edge_ids[edge_ids > 0])
# checker enclosed by the drawing (between the letters, under the shield) is backdrop too: a light component that
# holds flat squares of BOTH backdrop greys is the pattern, not a white or silver part of the drawing
if len(levels) == 2:
    lo = np.abs(grey - levels[0]) <= TOL
    hi = np.abs(grey - levels[1]) <= TOL
    sizes = np.bincount(lab.ravel())
    n_lo = np.bincount(lab[lo & (tint <= 3)].ravel(), minlength=sizes.size)
    n_hi = np.bincount(lab[hi & (tint <= 3)].ravel(), minlength=sizes.size)
    ids = np.nonzero((sizes >= 60) & (n_lo >= 12) & (n_hi >= 12) & ((n_lo + n_hi) >= 0.6 * sizes))[0]
    ids = ids[ids > 0]
    print('enclosed checker components:', len(ids), 'px', int(sizes[ids].sum()))
    back |= np.isin(lab, ids)
figure = ~back
# the figure's colour carried outward (nearest figure pixel), the backdrop colour carried inward (nearest pure pixel)
_, lf = cv2.distanceTransformWithLabels((~figure).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
ys, xs = np.nonzero(figure)
tab = np.zeros((lf.max() + 1, 3), np.float32); tab[lf[ys, xs]] = c[ys, xs]; F = tab[lf]
_, lb = cv2.distanceTransformWithLabels((~pure).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
ys, xs = np.nonzero(pure)
tab = np.zeros((lb.max() + 1, 3), np.float32); tab[lb[ys, xs]] = c[ys, xs]; W = tab[lb]
dv = W - F
a = np.clip(((W - c) * dv).sum(axis=2) / np.maximum((dv * dv).sum(axis=2), 1), 0, 1)
a[figure] = 1
a[pure & back] = 0
# the seams between checker squares (in-between greys) are backdrop too: only within a few px of the figure is a
# backdrop pixel an edge of the drawing
dist = cv2.distanceTransform((~figure).astype(np.uint8), cv2.DIST_L2, 5)
a[back & (dist > 2.5)] = 0
rgb = np.where(figure[:, :, None], c, F)
outim = np.dstack([rgb, a * 255]).round().astype(np.uint8)
Image.fromarray(outim, 'RGBA').save(out)
os.makedirs(prev, exist_ok=True)
for name, col in (('dark', (20, 12, 10)), ('red', (160, 30, 30)), ('green', (30, 110, 60))):
    bgim = Image.new('RGBA', im.size, col + (255,))
    bgim.alpha_composite(Image.fromarray(outim, 'RGBA'))
    bgim.save(os.path.join(prev, 'checker_on_%s.png' % name))
print('ok', out, 'backdrop share %.3f' % back.mean(), 'edge px', int(((a > 0) & (a < 1)).sum()))
