# Builds the site/ folder from sobor-1922.tpl.html: the page with the chapter text from
# scenario/glava1/*.txt inlined as data (format: scenario.py), and every sprite, backdrop and
# illustration written next to it as a WebP file under site/img/ (the page refers to them by
# relative address, so it also opens from the disk).
#   site/sobor-1922.html  the page as published (no doctype: the artifact host wraps it)
#   site/index.html       the same page with a doctype, for playing from the disk or a local server
#   site/files.json       {published path: file} of the pictures, for publishing them with the page
# The pictures were inlined into one file before 2026-10-13; the artifact limit (16 MB per page)
# ran out at 9.3 MB with half the backdrops still missing.
#
# Sprites live next to this folder: ../sprites/<character>/<set>_<emotion>.png
#   character  folder name, the id used in the script (kryukov, lunenko, ...)
#   set        first part of the file name; the organizer names the files himself,
#              usually after the character (kryukov_angry.png). A set is a "look" in the page.
#   emotion    whatever the file says: original, angry, scary, embarrased, ... (never renamed here)
# Only PNGs with a transparent background are taken; anything else in the folder
# (source photos, notes) is skipped.
#
# Emotions of one look are drawn at different scales, so a character would jump in size
# when the emotion changes. The build aligns them: within a look, the emotion with the
# largest figure is the reference, every other emotion is scaled up and shifted onto it
# (features matched with ORB, scale and shift from a RANSAC fit; rotation is not applied,
# figures stay upright). The canvas is widened so enlarged arms are not cut at the edge.
# The source files are never changed.
#
# Backgrounds: ../backgrounds/<name>.webp|png|jpg, where <name> is the backdrop id used
# in the script (barn, kremlin, ...). A backdrop without a file stays an SVG placeholder.
#
# Run:  .venv\Scripts\python.exe build.py
import io
import json
import math
import os
import re
import shutil
import zlib

import cv2
import numpy as np
import soundfile
from PIL import Image

import scenario

HERE = os.path.dirname(os.path.abspath(__file__))
SPRITES_DIR = os.path.join(HERE, '..', 'sprites')
BACKGROUNDS_DIR = os.path.join(HERE, '..', 'backgrounds')
LOGO = os.path.join(HERE, '..', 'logo', 'logo.png')  # the game's logo for the title card, cut off its white backdrop
CG_DIR = os.path.join(HERE, '..', 'cg')  # full-screen illustrations, id = file name = @cg in the script
CHAPTERS_DIR = os.path.join(HERE, '..', 'chapters')  # chapter covers for the shelf, id = file name = 4th field of `title`
BIOS_DIR = os.path.join(HERE, 'scenario', 'bio')  # character biographies, <who>.txt (scenario.load_bios)
SFX_DIR = os.path.join(HERE, '..', 'sfx')  # sound effects, <name>.mp3|ogg|wav: '@sfx name' in the script; paper, wire for the cards
MUSIC_DIR = os.path.join(HERE, '..', 'music')  # tracks, id = file name without .mp3 = @music in the script
# The version the players see, in the bottom right corner of the menus. Ours to change on releases worth naming, not
# on every publish (the artifact counts those): 1.0 is chapter I as released (15.10), 1.1, 1.2 for its fixes, 2.0
# with chapter II.
GAME_VERSION = '1.0'
# Release builds lock the scene list: a reader can jump only to scenes already reached, the rest are shown without
# titles (the organizer, 17.10: «чтобы людям было интересно, а то будут скипать»). False while the organizer tests.
SCENES_LOCKED = True
# The chapter shelf's music (a file in music/ without .mp3): plays on the shelf for every chapter until a chapter brings
# its own (5th field of `title`). Empty = silence after the title's march. The organizer sends the track (17.10).
SHELF_MUSIC = 'Пролог'
# «Поделиться» on the title (17.10). SHARE_URL: what is shared — empty = the page's own address (set the bot's t.me link
# when there is one). «Отзыв» and FEEDBACK_URL were removed 18.10: the organizer collects feedback another way.
SHARE_URL = 'https://t.me/AFM_Kryukov_bot'
CHAPTER = os.path.join(HERE, 'scenario', 'glava1')  # the chapter text, see scenario.py for the format
SITE = os.path.join(HERE, 'site')
# Aligning a set takes about a minute, so finished sprites are kept in .cache/sprites/<who>/<look>/<key>/ and reused
# while the key holds: the set's file names, sizes and times plus every alignment setting below. Bump ALIGN_VERSION
# when align_look or the sprite output changes in a way the key does not see.
CACHE_DIR = os.path.join(HERE, '.cache', 'sprites')
ALIGN_VERSION = 1
HEIGHT = 1100  # px; sprites are scaled to this height before saving
BG_WIDTH = 1920  # px; larger backgrounds are scaled down to this width
NAME = re.compile(r'([a-z]+(?:_[a-z]+)*)_([a-z]+)\.png')  # <set>_<emotion>.png; a set name may carry underscores (lunenko_povstanets)
SECOND_PLAN = 'other'  # sprites/other/<who>.png: second-plan characters, one picture each
ONE_PICTURE = re.compile(r'([a-z_]+)\.png')
# A second-plan picture is cut at the waist or the hips and stands high on the stage (its bottom at 82%, see BAND_TOP
# in the page); under a card the cut can show. The build used to fade out its lowest 22% — the organizer had the fade
# removed (15.10): the pictures are shown as drawn.
BG_NAME = re.compile(r'([a-z_]+)\.(webp|png|jpe?g)')
SIDE_PAD = 0.15  # fraction of the source width added on each side of an aligned sprite
TOP_PAD = 0.06  # fraction of the source height added above an aligned sprite, so enlarged hats are not cut
# The emotion whose size the whole look takes, when it is not simply the largest figure.
# Kryukov: the organizer picked the size of the pose then called angry (2026-10-12); he later
# renamed that file to unhappy and drew a new angry, so the reference follows the drawing.
REFERENCE = {('kryukov', 'kryukov'): 'unhappy'}
# Looks fitted on the top part of the figure only (a share of its height): the head and shoulders.
# By default the whole figure is matched, but when the poses turn the body differently the body
# scale and the head scale disagree. Ross: whole-figure fit made the heads of angry, sad and
# thinking 13-18% larger than the rest (hat brim width); fitted on the top quarter they stay within 5%.
FIT_TOP = {
    ('ross', 'ross'): 0.25,
    ('markov', 'markov'): 0.35,  # whole-figure fit: sad and thinking heads 12-15% larger; top third: within 6% but thinking
    ('pavlov', 'pavlov'): 0.45,  # whole figure: faces within 6.8%, the face centre moving 18 px; top 45%: 2.6%, steady
}
# Hand corrections on top of the fit, when no fit keeps an emotion's head the size of the rest (another camera
# angle, say). The emotion is scaled by the factor about its face (taken at FACE_AT of the drawing's height).
# Measure with the face widths of tools/measure_faces.py.
SCALE_FIX = {
    # angry and shocked faces 318 and 321 px against 296-300 in the others, the cap as much wider, whatever the fit
    ('glug', 'glug'): {'angry': 0.943, 'shocked': 0.935},
}
FACE_AT = 0.24
MIN_INLIERS = 80  # fewer matched features than this: the fit is not trusted, the emotion stays as drawn
SCALE_RANGE = (0.75, 1.4)  # a fitted scale outside this range is treated as a failed match

_orb = cv2.ORB_create(5000)
_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


def _features(im, top=1.0):
    a = np.array(im)
    gray = cv2.cvtColor(a[:, :, :3], cv2.COLOR_RGB2GRAY)
    mask = (a[:, :, 3] > 128).astype(np.uint8) * 255
    if top < 1.0:  # keep only the top share of the figure, measured from its highest to its lowest pixel
        rows = np.where(mask.any(axis=1))[0]
        mask[rows[0] + int((rows[-1] - rows[0]) * top):] = 0
    return _orb.detectAndCompute(gray, mask)


def fit(src, dst, top=1.0):
    """Scale and shift that put the figure of `src` onto the figure of `dst`: (s, tx, ty, inliers)."""
    ks, ds = _features(src, top)
    kd, dd = _features(dst, top)
    if ds is None or dd is None:
        return None
    matches = _matcher.match(ds, dd)
    if len(matches) < MIN_INLIERS:
        return None
    ps = np.float32([ks[m.queryIdx].pt for m in matches])
    pd = np.float32([kd[m.trainIdx].pt for m in matches])
    m, inl = cv2.estimateAffinePartial2D(ps, pd, method=cv2.RANSAC, ransacReprojThreshold=4)
    if m is None:
        return None
    inl = inl.ravel().astype(bool)
    s = math.hypot(m[0, 0], m[1, 0])  # rotation-free scale of the fitted similarity
    cs, cd = ps[inl].mean(axis=0), pd[inl].mean(axis=0)
    tx, ty = cd - s * cs  # shift that keeps the matched parts of the figure in place
    return s, float(tx), float(ty), int(inl.sum())


def align_look(who, look, emotions):
    """Brings all emotions of a look to the scale of its largest figure, on a widened canvas."""
    w, h = next(iter(emotions.values())).size
    pad, top = round(w * SIDE_PAD), round(h * TOP_PAD)
    canvas = (w + 2 * pad, h + top)
    names = sorted(emotions)
    ref = REFERENCE.get((who, look))
    top_share = FIT_TOP.get((who, look), 1.0)
    if ref not in emotions:
        base = 'original' if 'original' in emotions else names[0]
        # how big each figure is compared with the base emotion; the largest becomes the reference
        size = {base: 1.0}
        for e in names:
            if e != base:
                f = fit(emotions[e], emotions[base], top_share)
                size[e] = 1 / f[0] if f and f[3] >= MIN_INLIERS and SCALE_RANGE[0] <= f[0] <= SCALE_RANGE[1] else 1.0
        ref = max(names, key=lambda e: size[e])
    out = {}
    for e in names:
        im = emotions[e]
        s, tx, ty, note = 1.0, 0.0, 0.0, 'reference'
        if e != ref:
            f = fit(im, emotions[ref], top_share)
            if f and f[3] >= MIN_INLIERS and SCALE_RANGE[0] <= f[0] <= SCALE_RANGE[1]:
                s, tx, ty = f[0], f[1], f[2]
                note = 'x%.3f shift (%+.0f, %+.0f), %d matched points' % (s, tx, ty, f[3])
            else:
                note = 'NOT ALIGNED: no reliable match, left as drawn'
        k = SCALE_FIX.get((who, look), {}).get(e)
        if k:  # scale about the face, which sits at (w/2, FACE_AT*h) in the reference's coordinates
            px, py = w / 2, FACE_AT * h
            s, tx, ty = s * k, k * tx + (1 - k) * px, k * ty + (1 - k) * py
            note += ', hand fix x%.3f' % k
        mat = np.float32([[s, 0, tx + pad], [0, s, ty + top]])
        a = cv2.warpAffine(np.array(im), mat, canvas, flags=cv2.INTER_LANCZOS4,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        if a[0, :, 3].max() > 128:
            note += ' (WARNING: head touches the top edge)'
        out[e] = Image.fromarray(a, 'RGBA')
        print('align', who, look, e, note)
    return out, top / canvas[1]


def webp(im, quality):
    buf = io.BytesIO()
    im.save(buf, 'WEBP', quality=quality, method=6)
    return buf.getvalue()


def write_site(data, rel):
    """Writes a file under site/; returns its address for the page and its size. The address carries a hash of the
    content, so a browser that kept an older picture under the same name fetches the new one."""
    path = os.path.join(SITE, *rel.split('/'))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        f.write(data)
    return '%s?v=%08x' % (rel, zlib.crc32(data)), len(data)


def save_image(im, quality, rel):
    return write_site(webp(im, quality), rel)


def load_logo():
    """The logo next to the page (img/logo.webp); its address, or '' when there is none."""
    if not os.path.isfile(LOGO):
        return ''
    im = Image.open(LOGO).convert('RGBA')
    if im.width > 1400:
        im = im.resize((1400, round(im.height * 1400 / im.width)), Image.LANCZOS)
    url, size = save_image(im, 88, 'img/logo.webp')
    print('logo', im.size, size, 'bytes')
    return url


def load_backgrounds(folder=BACKGROUNDS_DIR, kind='background', sub='bg', only=None):
    """`only`: the names the script uses; the rest stay in the source folder and are not copied to the site (16.10)."""
    backgrounds = {}
    if not os.path.isdir(folder):
        return backgrounds
    for name in sorted(os.listdir(folder)):
        m = BG_NAME.fullmatch(name)
        if not m:
            continue
        if only is not None and m.group(1) not in only:
            print(kind, m.group(1), 'not in the script: not shipped')
            continue
        im = Image.open(os.path.join(folder, name)).convert('RGB')
        if im.width > BG_WIDTH:
            im = im.resize((BG_WIDTH, round(im.height * BG_WIDTH / im.width)), Image.LANCZOS)
        backgrounds[m.group(1)], size = save_image(im, 82, 'img/%s/%s.webp' % (sub, m.group(1)))
        print(kind, m.group(1), im.size, size, 'bytes')
    return backgrounds


TRANSLIT = dict(zip('абвгдеёжзийклмнопрстуфхцчшщъыьэюя',
                    ['a', 'b', 'v', 'g', 'd', 'e', 'e', 'zh', 'z', 'i', 'y', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't', 'u',
                     'f', 'kh', 'ts', 'ch', 'sh', 'shch', '', 'y', '', 'e', 'yu', 'ya']))


def latin(name):
    """A file name the web can carry without escaping: transliterated, lower case, words joined by _."""
    s = ''.join(TRANSLIT.get(c, c) for c in name.lower())
    return re.sub(r'[^a-z0-9]+', '_', s).strip('_') or 'track'


def loop_points(path):
    """Where the music of a track begins and ends, in seconds: the silence it opens with and the silent or faded tail
    after it are left out, so a loop joins music to music. Loud = within 18 dB of the track's median loudness."""
    x, sr = soundfile.read(path, dtype='float32', always_2d=True)
    x = x.mean(axis=1)
    hop = int(sr * 0.1)
    n = len(x) // hop
    db = 20 * np.log10(np.sqrt((x[:n * hop].reshape(n, hop) ** 2).mean(axis=1)) + 1e-9)
    loud = np.where(db > np.median(db) - 18)[0]
    return round(max(0.0, loud[0] * 0.1 - 0.05), 2), round((loud[-1] + 1) * 0.1, 2)


def load_sfx():
    """Sound effects copied as they are: { name: url }. CC0 recordings picked 17.10 (sources in sfx/SOURCES.txt)."""
    out = {}
    if not os.path.isdir(SFX_DIR):
        return out
    for name in sorted(os.listdir(SFX_DIR)):
        stem, ext = os.path.splitext(name)
        if ext.lower() not in ('.mp3', '.ogg', '.wav'):
            continue
        with open(os.path.join(SFX_DIR, name), 'rb') as f:
            data = f.read()
        out[stem], size = write_site(data, 'audio/sfx_%s%s' % (latin(stem), ext.lower()))
        print('sfx', stem, size, 'bytes')
    return out


MUSIC_VBR = 0.4  # libsndfile/LAME VBR quality: about 130 kbps (the organizer's files are ~180–260 kbps; 17.10: lighter for slow lines)
MUSIC_CACHE = os.path.join(HERE, '.cache', 'music')


def encode_music(path):
    """The track re-encoded as a lighter mp3 (VBR, MUSIC_VBR), cached by the source file's size and date."""
    st = os.stat(path)
    key = '%08x' % zlib.crc32(('%s|%d|%d|%s' % (os.path.basename(path), st.st_size, st.st_mtime_ns, MUSIC_VBR)).encode('utf-8'))
    os.makedirs(MUSIC_CACHE, exist_ok=True)
    cached = os.path.join(MUSIC_CACHE, key + '.mp3')
    if os.path.isfile(cached):
        with open(cached, 'rb') as f:
            return f.read()
    x, sr = soundfile.read(path, dtype='float32', always_2d=True)
    buf = io.BytesIO()
    with soundfile.SoundFile(buf, 'w', samplerate=sr, channels=x.shape[1], format='MP3', compression_level=MUSIC_VBR, bitrate_mode='VARIABLE') as f:
        f.write(x)
    data = buf.getvalue()
    with open(cached, 'wb') as f:
        f.write(data)
    return data


def load_music(only=None):
    """Copies the tracks next to the page as they are (no re-encoding); { name: {u: url, s: start, e: end} },
    name = file name, s/e = where the music begins and ends (loop_points). `only`: the names the script uses — the
    other tracks of music/ stay home (16.10: six unused tracks were 20 MB of a 66 MB site)."""
    music = {}
    if not os.path.isdir(MUSIC_DIR):
        return music
    for name in sorted(os.listdir(MUSIC_DIR)):
        stem, ext = os.path.splitext(name)
        if ext.lower() not in ('.mp3', '.ogg', '.m4a'):
            continue
        if only is not None and stem not in only:
            print('music', stem, 'not in the script: not shipped')
            continue
        data = encode_music(os.path.join(MUSIC_DIR, name))
        url, size = write_site(data, 'audio/%s.mp3' % latin(stem))
        start, end = loop_points(os.path.join(MUSIC_DIR, name))
        music[stem] = {'u': url, 's': start, 'e': end}
        print('music', stem, size, 'bytes, music from %.1f to %.1f s' % (start, end))
    return music


def look_key(who, look, paths):
    """What a built set depends on: its files and the alignment settings."""
    stats = [(os.path.basename(p), os.path.getsize(p), os.stat(p).st_mtime_ns) for p in paths]
    conf = [ALIGN_VERSION, stats, REFERENCE.get((who, look)), FIT_TOP.get((who, look)), SCALE_FIX.get((who, look)),
            SIDE_PAD, TOP_PAD, HEIGHT, MIN_INLIERS, SCALE_RANGE, FACE_AT]
    return '%08x' % zlib.crc32(json.dumps(conf, sort_keys=True, default=str).encode('utf-8'))


def build_look(who, look, items, cache):
    """Aligns and scales one set, writes its WebPs and meta.json into `cache`; returns the meta."""
    emotions = {}
    for emotion, path in items:
        im = Image.open(path).convert('RGBA')
        if im.getchannel('A').getextrema()[0] == 255:
            print('skip (no transparency):', who + '/' + os.path.basename(path))
            continue
        emotions[emotion] = im
    pad_top = 0.0  # share of the canvas height added above the drawing; the page offsets it
    if len(emotions) > 1:
        emotions, pad_top = align_look(who, look, emotions)
    shutil.rmtree(os.path.dirname(cache), ignore_errors=True)  # older builds of this set
    os.makedirs(cache)
    meta = {'ar': 1.0, 'pt': round(pad_top, 4), 'emotions': []}
    for emotion, im in sorted(emotions.items()):
        im = im.resize((round(im.width * HEIGHT / im.height), HEIGHT), Image.LANCZOS)
        meta['ar'] = round(im.width / im.height, 4)
        with open(os.path.join(cache, emotion + '.webp'), 'wb') as f:
            f.write(webp(im, 80))
        meta['emotions'].append(emotion)
    with open(os.path.join(cache, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f)
    return meta


def sprite_sets():
    """[(who, look, [(emotion, path)])]: the sets sprites/<who>/<look>_<emotion>.png, and the second-plan characters
    of sprites/other/, one picture each: <who>.png is his only look, worn for every emotion. Sources lying next to
    them are skipped: the organizer's .jpeg on a light backdrop (cut with tools/cut_white.py) and <who>_ref.png, the
    original kept when a picture was cleaned (tools/defringe.py)."""
    sets = []
    for who in sorted(os.listdir(SPRITES_DIR)):
        folder = os.path.join(SPRITES_DIR, who)
        if not os.path.isdir(folder):
            continue
        if who == SECOND_PLAN:
            for name in sorted(os.listdir(folder)):
                m = ONE_PICTURE.fullmatch(name)
                if m and not m.group(1).endswith('_ref'):
                    sets.append((m.group(1), m.group(1), [('original', os.path.join(folder, name))]))
            continue
        looks = {}
        for name in sorted(os.listdir(folder)):
            m = NAME.fullmatch(name)
            if m:
                looks.setdefault(m.group(1), []).append((m.group(2), os.path.join(folder, name)))
        sets += [(who, look, items) for look, items in looks.items()]
    return sets


def load_sprites():
    sprites = {}
    for who, look, items in sprite_sets():
        cache = os.path.join(CACHE_DIR, who, look, look_key(who, look, [p for _, p in items]))
        meta_path = os.path.join(cache, 'meta.json')
        if os.path.isfile(meta_path):
            with open(meta_path, encoding='utf-8') as f:
                meta = json.load(f)
            print('sprite', who, look, 'from cache:', ', '.join(meta['emotions']))
        else:
            meta = build_look(who, look, items, cache)
        if not meta['emotions']:
            continue
        entry = sprites.setdefault(who, {})[look] = {'ar': meta['ar'], 'pt': meta['pt'], 'emotions': {}}
        for emotion in meta['emotions']:
            with open(os.path.join(cache, emotion + '.webp'), 'rb') as f:
                data = f.read()
            entry['emotions'][emotion], size = write_site(data, 'img/sprites/%s/%s_%s.webp' % (who, look, emotion))
            print('sprite', who, look, emotion, size, 'bytes')
    return sprites


def load_script(html, backgrounds, sprites, cgs, music):
    """The chapter text, checked against the characters of the template; stops the build on mistakes."""
    steps, state = scenario.load_chapter(CHAPTER)
    characters = set(re.findall(r'^\s{4}([a-z_]+):\s*\{\s*n:', html, re.M))
    problems = scenario.check(steps, characters, backgrounds)
    for p in problems:
        print('script:', p)
    if problems:
        raise SystemExit('fix the script first')
    # an emotion the script names but no file has (often a file renamed by the organizer): the page shows original
    asked = {}
    for s in steps:
        for who, e in (s.get('e') or {}).items():
            have = {emo for look in sprites.get(who, {}).values() for emo in look['emotions']}
            if have and e not in have:
                asked.setdefault((who, e), []).append(s.get('src', '-'))
    for (who, e), where in sorted(asked.items()):
        print('emotion without a file (original shown): %s.%s at #%s' % (who, e, ', #'.join(map(str, where[:6]))))
    used = sorted({s['bg'] for s in steps if s.get('bg')})
    for bg in used:
        if bg not in backgrounds:
            print('background missing (placeholder shown):', bg)
    for cg in sorted({s['cg'] for s in steps if s.get('cg')}):
        if cg not in cgs:
            print('illustration missing (placeholder shown): cg/%s.png' % cg)
    for track in sorted({s['music'] for s in steps if s.get('music')} | ({state['title']['music']} - {''})):
        if track not in music:
            print('music missing (silence instead): music/%s.mp3' % track)
    # Each step gets a fingerprint of its content: saves and "already read" marks refer to it, so they
    # survive edits elsewhere in the script. `ver` changes whenever anything in the chapter changes.
    for s in steps:
        s['h'] = '%08x' % zlib.crc32(json.dumps({k: v for k, v in s.items() if k not in ('p', 'h')}, ensure_ascii=False, sort_keys=True).encode('utf-8'))
    ver = '%08x' % zlib.crc32(json.dumps(steps, ensure_ascii=False, sort_keys=True).encode('utf-8'))
    print('script', len(steps), 'steps,', len(state['scenes']), 'scenes, version', ver)
    return {'title': state['title'], 'scenes': state['scenes'], 'steps': steps, 'ver': ver}


def main():
    for sub in ('img', 'audio'):  # built output only: pictures and tracks dropped from the sources go too
        shutil.rmtree(os.path.join(SITE, sub), ignore_errors=True)
    # only what the chapter names goes to the site: its backdrops, illustrations and tracks (plus the title's)
    steps, state = scenario.load_chapter(CHAPTER)
    used_bg = {s['bg'] for s in steps if s.get('bg')} | {'main_menu'}
    used_cg = {s['cg'] for s in steps if s.get('cg')}
    used_music = {s['music'] for s in steps if s.get('music')} | {state['title']['music'], state['title']['theme'], SHELF_MUSIC} - {''}
    sprites, backgrounds, cgs = load_sprites(), load_backgrounds(only=used_bg), load_backgrounds(CG_DIR, 'illustration', 'cg', only=used_cg)
    music = load_music(only=used_music)
    sfx = load_sfx()
    for s in steps:
        if s.get('sfx') and s['sfx'] not in sfx:
            print('sound missing (silence instead): sfx/%s.mp3' % s['sfx'])
    covers = load_backgrounds(CHAPTERS_DIR, 'cover', 'chapters')
    if state['title']['cover'] and state['title']['cover'] not in covers:
        print('cover missing (blank card): chapters/%s.png' % state['title']['cover'])
    with open(os.path.join(HERE, 'sobor-1922.tpl.html'), encoding='utf-8') as f:
        html = f.read()
    bios = scenario.load_bios(BIOS_DIR)
    characters = set(re.findall(r'^\s{4}([a-z_]+):\s*\{\s*n:', html, re.M))
    for who in sorted(bios):
        if who not in characters:
            raise SystemExit('bio for unknown character: scenario/bio/%s.txt' % who)
    print('bios', ', '.join(sorted(bios)) or '(none)')
    for key in ("'__SPRITES__'", "'__BACKGROUNDS__'", "'__CGS__'", "'__MUSIC__'", "'__SCRIPT__'", "'__VERSION__'", "'__LOGO__'", "'__LOCKED__'", "'__COVERS__'", "'__SHELF_MUSIC__'", "'__BIOS__'", "'__LINKS__'", "'__SFX__'"):
        if html.count(key) != 1:
            raise SystemExit('template must contain the ' + key + ' placeholder exactly once')
    script = load_script(html, backgrounds, sprites, cgs, music)
    html = (html.replace("'__SPRITES__'", json.dumps(sprites)).replace("'__BACKGROUNDS__'", json.dumps(backgrounds))
            .replace("'__CGS__'", json.dumps(cgs)).replace("'__MUSIC__'", json.dumps(music, ensure_ascii=False)).replace("'__SCRIPT__'", json.dumps(script, ensure_ascii=False))
            .replace("'__VERSION__'", json.dumps('версия ' + GAME_VERSION, ensure_ascii=False))
            .replace("'__LOCKED__'", json.dumps(SCENES_LOCKED)).replace("'__COVERS__'", json.dumps(covers)).replace("'__SHELF_MUSIC__'", json.dumps(SHELF_MUSIC, ensure_ascii=False)).replace("'__BIOS__'", json.dumps(bios, ensure_ascii=False))
            .replace("'__LINKS__'", json.dumps({'share': SHARE_URL})).replace("'__SFX__'", json.dumps(sfx))
            .replace("'__LOGO__'", json.dumps(load_logo())))
    with open(os.path.join(SITE, 'sobor-1922.html'), 'w', encoding='utf-8') as f:
        f.write(html)
    with open(os.path.join(SITE, 'index.html'), 'w', encoding='utf-8') as f:
        f.write('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
                '</head><body>' + html + '</body></html>')
    files, total = {}, 0
    for sub in ('img', 'audio'):
        for root, _, names in os.walk(os.path.join(SITE, sub)):
            for name in sorted(names):
                path = os.path.join(root, name)
                files[os.path.relpath(path, SITE).replace(os.sep, '/')] = path
                total += os.path.getsize(path)
    with open(os.path.join(SITE, 'files.json'), 'w', encoding='utf-8') as f:
        json.dump(dict(sorted(files.items())), f, ensure_ascii=False, indent=1)
    print('built', SITE, ': page', len(html.encode('utf-8')), 'bytes,', len(files), 'pictures and tracks', total, 'bytes')


if __name__ == '__main__':
    main()
