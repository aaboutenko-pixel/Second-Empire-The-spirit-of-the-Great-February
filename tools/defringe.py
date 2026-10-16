# Takes a light fringe off a figure that was cut out of a light backdrop: its outermost pixels keep a trace of the
# backdrop and draw a thin pale outline on a dark scene (Antonov-Ovseenko, 14.10). Within BAND px of the outline a
# pixel lighter than the figure's own colour there (carried out from inside) takes that colour; the outermost ring of
# pixels is feathered a little.
# Run from vn-demo:  .venv\Scripts\python.exe -I tools\defringe.py <in.png> <out.png> <preview.png> [band_px]
# The preview shows the result over a dark backdrop: look at hair, shoulders and sleeves before using it.
import sys

import cv2
import numpy as np
from PIL import Image

src, out, prev = sys.argv[1], sys.argv[2], sys.argv[3]
BAND = int(sys.argv[4]) if len(sys.argv) > 4 else 3


def spread(values, weight, sigmas):
    """Values known where weight is 1, carried over the rest: normalised blur, finer scales first."""
    res = np.zeros_like(values)
    have = np.zeros(weight.shape, bool)
    for sigma in sigmas:
        k = cv2.GaussianBlur(weight, (0, 0), sigma)
        f = cv2.GaussianBlur(values * weight[:, :, None], (0, 0), sigma)
        take = ~have & (k > 0.02)
        res[take] = f[take] / k[take, None]
        have |= take
    return res


im = np.array(Image.open(src).convert('RGBA')).astype(np.float32)
rgb, a = im[:, :, :3], im[:, :, 3] / 255
solid = (a > 0.5).astype(np.uint8)
inner = cv2.erode(solid, np.ones((2 * BAND + 1, 2 * BAND + 1), np.uint8)).astype(bool)
fill = spread(rgb, inner.astype(np.float32), (2, 4, 8, 16))

near = (a > 0) & ~inner  # the outline band and whatever half-transparent pixels lie outside it
lum = lambda c: c[..., 0] * 0.299 + c[..., 1] * 0.587 + c[..., 2] * 0.114
t = np.clip((lum(rgb) - lum(fill) - 6) / 30, 0, 1) * near  # how much lighter than the figure there
rgb = rgb * (1 - t[:, :, None]) + fill * t[:, :, None]

ring = solid.astype(bool) & ~cv2.erode(solid, np.ones((3, 3), np.uint8)).astype(bool)
a = np.where(ring, a * 0.6, a)
a = np.where(solid.astype(bool), a, a * 0.5)  # stray half-transparent pixels outside the figure fade further

res = np.dstack([rgb, a * 255]).round().clip(0, 255).astype(np.uint8)
Image.fromarray(res, 'RGBA').save(out, optimize=True)

A = (a[:, :, None]).astype(np.float32)
comp = rgb * A + np.array((20, 22, 28), np.float32) * (1 - A)
Image.fromarray(comp.round().astype(np.uint8)).save(prev)
print('ok', out, 'band px %d, recoloured px %d' % (int(near.sum()), int((t > 0.05).sum())))
