# Cuts a drawing off a pure white backdrop, keeping every white or light part that is enclosed by the drawing
# (the white stripe of a flag, a white deer on a blue shield). Only the white region connected to the picture
# border is removed; along its edge the drawing's own colour is carried outward and the alpha is solved from
# C = a*F + (1-a)*W. Usage: cut_logo.py <src> <out.png> <preview_dir>
import os
import sys

import cv2
import numpy as np
from PIL import Image

LIGHT = 36   # a pixel closer than this to white may be backdrop (border-connected); darker edges block the fill
W = 255.0

src, out, prev = sys.argv[1], sys.argv[2], sys.argv[3]
im = Image.open(src).convert('RGB')
c = np.array(im).astype(np.float32)
dist = (W - c).max(axis=2)  # how far from white, by the darkest channel
light = (dist < LIGHT).astype(np.uint8)
n, lab = cv2.connectedComponents(light, connectivity=4)
edge_ids = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
back = np.isin(lab, edge_ids[edge_ids > 0])  # the backdrop: light and reaching the border
figure = ~back
# the figure's colour carried outward: each backdrop pixel takes the colour of the nearest figure pixel
inv = figure.astype(np.uint8)  # zero where the figure is, so the transform finds the nearest figure pixel
_, labels = cv2.distanceTransformWithLabels(1 - inv, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
ys, xs = np.nonzero(figure)
by_label = np.zeros((labels.max() + 1, 3), np.float32)
by_label[labels[ys, xs]] = c[ys, xs]
F = by_label[labels]
# alpha: projection of (W - C) onto (W - F); pure backdrop gives 0, pure figure colour 1
d = W - F
num = ((W - c) * d).sum(axis=2)
den = (d * d).sum(axis=2)
a = np.where(den > 1, num / np.maximum(den, 1), 0)
a = np.clip(a, 0, 1)
a[figure] = 1
a[back & (dist <= 4)] = 0
rgb = np.where(figure[:, :, None], c, F)
outim = np.dstack([rgb, a * 255]).round().astype(np.uint8)
Image.fromarray(outim, 'RGBA').save(out)
os.makedirs(prev, exist_ok=True)
for name, col in (('dark', (20, 12, 10)), ('red', (160, 30, 30)), ('green', (30, 110, 60))):
    bg = Image.new('RGBA', im.size, col + (255,))
    bg.alpha_composite(Image.fromarray(outim, 'RGBA'))
    bg.save(os.path.join(prev, 'mine_on_%s.png' % name))
print('ok', out, 'backdrop share %.3f' % back.mean(), 'edge px', int(((a > 0) & (a < 1)).sum()))
