# Cut a figure off a plain light backdrop (white, or light grey with a vignette): soft alpha along the edge, no light
# fringe. For the second-plan pictures the organizer draws on a light backdrop
# (sprites/other/tkachenko_ref.jpeg -> sprites/other/tkachenko.png).
# Run from vn-demo:  .venv\Scripts\python.exe -I tools\cut_white.py <src> <out.png> <preview_dir>
# preview_dir gets the cut-out over dark, red and green: look at the edges there before using it.
#
# The backdrop colour B is estimated at every pixel from the light, colourless pixels reaching the picture border
# (a grey backdrop is rarely even). dist = how far a pixel is from B. Pixels close to B (dist < LIGHT) that connect
# to the border are backdrop or edge; everything else is the figure, including light spots inside it (the hilt).
# On the edge the figure's own colour is carried outward from the solid figure, and alpha says how much of that
# colour, mixed with the backdrop, makes the pixel: C = a*F + (1-a)*B.
import os
import sys

import cv2
import numpy as np
from PIL import Image

LIGHT = 45     # below this distance from the backdrop a pixel may belong to it
CLEAR = 7      # at or below this it is the backdrop for sure (JPEG noise and the estimate of B stay under it)


def border_parts(mask):
    """The parts of a mask that reach the picture border."""
    n, lab = cv2.connectedComponents(mask.astype(np.uint8), connectivity=4)
    edge = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    return np.isin(lab, edge[edge > 0]), n, lab


def spread(values, weight, sigmas):
    """Values known where weight is 1, carried over the rest: normalised blur, finer scales first."""
    out = np.zeros_like(values)
    have = np.zeros(weight.shape, bool)
    for sigma in sigmas:
        k = cv2.GaussianBlur(weight, (0, 0), sigma)
        f = cv2.GaussianBlur(values * weight[:, :, None], (0, 0), sigma)
        take = ~have & (k > 0.02)
        out[take] = f[take] / k[take, None]
        have |= take
    out[~have] = values[weight > 0].mean(axis=0) if weight.any() else 128
    return out


src, out, prev = sys.argv[1], sys.argv[2], sys.argv[3]
c = np.array(Image.open(src).convert('RGB')).astype(np.float32)
h, w = c.shape[:2]

# the backdrop: light, colourless, reaching the border; its colour spread smoothly under the figure
seed = border_parts((c.min(axis=2) > 190) & (c.max(axis=2) - c.min(axis=2) < 16))[0]
seed = cv2.erode(seed.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(np.float32)  # away from the edge
back = spread(c, seed, (12, 30, 80, 200))
dist = np.abs(c - back).max(axis=2)

outer, n, lab = border_parts(dist < LIGHT)
# backdrop seen through a gap the figure closes (between the elbow and the belt): a light patch that is mostly
# backdrop and not small; light spots of the figure itself (the hilt) are small and not backdrop-coloured
for i in range(1, n):
    part = lab == i
    if not outer[part].any():
        clear = (dist[part] <= CLEAR).sum()
        if clear > 300 and clear > 0.5 * part.sum():
            outer |= part
solid = ~outer
# the outermost two pixels of the figure are still mixed with the backdrop (a grey line along the black papakha):
# they get their alpha like the rest of the edge
band = solid & cv2.dilate(outer.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
inner = solid & ~band

# the figure's colour carried out over the edge band
fill = spread(c, inner.astype(np.float32), (2, 4, 8, 16))

# alpha on the edge: how far the pixel went from the backdrop towards the figure colour
df = back - fill
a = np.clip(((back - c) * df).sum(axis=2) / np.maximum((df * df).sum(axis=2), 1), 0, 1)
a[inner] = 1
a[outer & (dist <= CLEAR)] = 0
# far from the figure (smoke over the backdrop) keep only a faint trace of what is there
near = cv2.dilate(solid.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
smoke = outer & ~near & (dist > CLEAR)
k, slab, st, _ = cv2.connectedComponentsWithStats(smoke.astype(np.uint8))
smoke &= np.isin(slab, [i for i in range(1, k) if st[i, cv2.CC_STAT_AREA] >= 40])  # lone specks are noise, not smoke
a[outer & ~near & ~smoke] = 0
a[smoke] = np.clip(dist[smoke] / 60, 0, 0.35)

# edge colour: unmixed from the backdrop where the figure dominates the pixel, the carried colour where it does not
A = a[:, :, None]
unmix = np.clip((c - (1 - A) * back) / np.maximum(A, 1e-3), 0, 255)
rgb = np.where(inner[:, :, None], c, np.where(A > 0.6, unmix, fill))
rgb[smoke] = unmix[smoke]  # smoke: its own colour unmixed from the backdrop

rgba = np.dstack([rgb, a * 255]).round().clip(0, 255).astype(np.uint8)
os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
Image.fromarray(rgba, 'RGBA').save(out, optimize=True)

os.makedirs(prev, exist_ok=True)
for name, col in (('on_dark', (24, 26, 32)), ('on_red', (200, 40, 40)), ('on_green', (40, 160, 70))):
    comp = rgb * A + np.array(col, np.float32) * (1 - A)
    Image.fromarray(comp.round().astype(np.uint8)).save(os.path.join(prev, name + '.png'))
print('ok', out, 'figure share %.3f, edge px %d, smoke px %d, backdrop %s..%s' % (
    (a > .5).mean(), int(((a > 0) & (a < 1)).sum()), int((smoke & (a > 0)).sum()),
    back.min(axis=(0, 1)).round().astype(int).tolist(), back.max(axis=(0, 1)).round().astype(int).tolist()))
