# Measures every character look on the stage the way the page places it, so sizes are set by numbers,
# not by eye: face width in px on a 720 px stage and the height of the face centre, per emotion and
# as a median. Also suggests the template's h (face to TARGET_FACE px) and top (face centre to
# TARGET_CENTRE %). The emotions are aligned exactly as build.py aligns them.
#
# Face = the largest skin-coloured blob (YCrCb, Cr 138-175, Cb 85-130) in the top 42% of the canvas;
# a hand touching the face joins the blob, so single emotions can read wide: trust the median.
#
# Run from vn-demo:  .venv\Scripts\python.exe -I tools\measure_faces.py [character ...]
import contextlib
import io
import os
import re
import statistics
import sys

import cv2
import numpy as np
from PIL import Image

VN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, VN)
import build  # noqa: E402

STAGE = 720  # px, stage height the numbers are given for
TARGET_FACE = 137  # px, what the cast measures now
TARGET_CENTRE = 28.0  # % of the stage height, median face centre of the cast
BAND_TOP = 80  # same as the page: a drawing must reach BAND_TOP + 2


def placements():
    tpl = open(os.path.join(VN, 'sobor-1922.tpl.html'), encoding='utf-8').read()
    return {m.group(1): (int(m.group(2)), int(m.group(3)))
            for m in re.finditer(r"(\w+): \{ r: '[^']*', top: (\d+), h: (\d+)", tpl)}


def face(a):
    cut = int(a.shape[0] * .42)
    ycc = cv2.cvtColor(np.ascontiguousarray(a[:cut, :, :3]), cv2.COLOR_RGB2YCrCb)
    mask = ((ycc[:, :, 1] >= 138) & (ycc[:, :, 1] <= 175) & (ycc[:, :, 2] >= 85) & (ycc[:, :, 2] <= 130)
            & (a[:cut, :, 3] > 128)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, _, st, cen = cv2.connectedComponentsWithStats(mask)
    if n < 2:
        return None
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    return st[i, cv2.CC_STAT_WIDTH], cen[i][1]


def main():
    place = placements()
    only = sys.argv[1:]
    for who, look, items in build.sprite_sets():
        if only and who not in only:
            continue
        emotions = {e: Image.open(p).convert('RGBA') for e, p in items}
        src_h = next(iter(emotions.values())).size[1]
        with contextlib.redirect_stdout(io.StringIO()):
            aligned = build.align_look(who, look, emotions)[0] if len(emotions) > 1 else emotions
        pad = round(src_h * build.TOP_PAD) if len(emotions) > 1 else 0
        top, h = place.get(look, (0, 100))
        top = max(top, BAND_TOP + 2 - h)  # the page lowers a drawing that does not reach the band
        widths, centres = {}, []
        for e, im in sorted(aligned.items()):
            f = face(np.array(im))
            if f:
                widths[e] = f[0] * h / 100 * STAGE / src_h
                centres.append(top + (f[1] - pad) / src_h * h)
        w, c = statistics.median(widths.values()), statistics.median(centres)
        new_h = round(h * TARGET_FACE / w)
        # face centre in % of the drawing does not depend on h, so the new top follows directly
        new_top = round(TARGET_CENTRE - (c - top) / h * new_h)
        print('%s/%s  top=%d h=%d  face %.0f px, centre %.1f%%  -> h %d, top %d' % (
            who, look, top, h, w, c, new_h, max(new_top, BAND_TOP + 2 - new_h)))
        print('    ' + '  '.join('%s %.0f' % (e, v) for e, v in widths.items()))


if __name__ == '__main__':
    main()
