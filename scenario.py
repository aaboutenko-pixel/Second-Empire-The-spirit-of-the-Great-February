# Reads a chapter written as plain text (scenario/<chapter>/*.txt, files in name order) into the
# list of steps the page plays. The format is meant to be written and edited by hand.
#
#   // comment                                 ignored
#   title Глава I · Собор | ноябрь 1922 — сентябрь 1923 | Слава РИА РП
#                                              chapter name | dates | music of the title screen (optional)
#   == kremlin | Кремль: съезд | Кремль, 16 ноября 1922
#                                              scene: jump id, entry in the scene list, place shown on top
#   @bg kremlin_zal                            backdrop (backgrounds/<id>.png); holds until changed
#   @on taburetov kryukov ross                 who stands on stage, left to right; "@on" alone clears it
#   @x ross=64                                 move a character off its slot (in % of the stage width)
#   @e ross=sad kryukov=angry                  emotions of the people on stage for the next line
#   @id canon                                  jump target name for the next line
#   @role kryukov=император                    caption under the name from the next line on; holds
#   @fx dust                                   effect over the scene: dust, snow, rain, embers, smoke, none;
#                                              holds until changed, a new scene starts without one
#   @shake  @flash                             one-off on the next line: the screen shakes / flashes white
#   @cg vzryv                                  a full-screen illustration (cg/<id>.png) instead of the backdrop
#                                              and the characters; "@cg" alone, a new @bg or a new scene ends it
#   @music Судьба Кашубского                   music from the next line on (music/<name>.mp3, the organizer's
#                                              file name), looped; holds across scenes; "@music" alone = silence
#
#   757 ross.angry: Текст реплики.             a line: message number, speaker id, emotion
#   757 ross: Текст.                           same emotion as before
#   757 > Текст описания.                      narration
#   + ross: Текст.                             "+" instead of the number: the line is not in the game log
#   ~ ross: Текст.                             "~": the road not taken (a what-if branch)
#   > Текст.                                   narration without a source
#
# Cards are blocks: a header line, then body lines up to the next blank line. In any card a line
# "= текст" is the caption shown in the text box under the card instead of "click to go on".
#   paper 66 | Ведомости России | 3 марта 1918 г. | Заголовок
#   текст абзаца                               (each line is a paragraph; "> " starts a quote)
#   wire 567 | от кого | кому                  lines are the tape strips
#   letter 4060 | обращение | подпись          lines are paragraphs
#   doc 973 | Название документа               lines are the articles
#   ? 786 Что ответит Росс?                    a choice; lines "* текст -> id"
#   fork 792 | Так в игре не было              lines are paragraphs ("> " = note), "* текст -> id" buttons
#   end | дата | Заголовок                     chapter or scene end card; lines as in fork
#   credits | Заголовок | Конец пролога        the cast roll (16.10; the third field goes on the curtain before it): lines "who | Полное имя | player" (a character
#                                              id: his portrait, surname and caption come from the page; the full
#                                              name may be left out), "текст | имя" (any other left side: a plain
#                                              row), "# Заголовок" (a heading); no buttons: it ends on the title screen
#
# A source number may be followed by "+": "757+ ross: ..." means the line is cut down or joined
# from several messages; it still links to 757.
import os
import re

SPEECH = re.compile(r'(?P<src>\d+\+?|\+|~)?\s*(?:(?P<who>[a-z_]+)(?:\.(?P<emo>[a-z]+))?:|>)\s*(?P<t>.+)$')
CARD = re.compile(r'(paper|wire|letter|doc|fork|end|credits)\b\s*(?P<src>\d+)?\s*(?:\|\s*(?P<rest>.*))?$')
CHOICE = re.compile(r'\?\s*(?P<src>\d+)?\s*(?P<ask>.+)$')
BUTTON = re.compile(r'\*\s*(?P<t>.+?)\s*->\s*(?P<go>[a-z_0-9]+)$')


class ScenarioError(Exception):
    pass


def _source(step, src):
    if not src:
        return
    if src == '+':
        step['added'] = True
    elif src == '~':
        step['fork'] = True
    else:
        step['src'] = int(src.rstrip('+'))


def parse(text, fname='', state=None):
    """Steps of one file. `state` carries scene/place between files of one chapter."""
    state = state if state is not None else {'p': '', 'scenes': [], 'title': None}
    steps, pending = [], {}
    lines = text.splitlines()
    i = 0

    def fail(msg):
        raise ScenarioError('%s:%d: %s' % (fname, i + 1, msg))

    def take(step):
        step['p'] = state['p']
        for k in ('bg', 'on', 'x', 'id', 'role', 'fx', 'cg', 'once', 'music', 'sfx'):
            if k in pending:
                step[k] = pending.pop(k)
        if 'e' in pending:
            step['e'] = dict(pending.pop('e'), **(step.get('e') or {}))
        if 'scene' in pending:
            step['id'] = step.get('id') or pending['scene']
            pending.pop('scene')
        steps.append(step)

    def starts_statement(line):
        """A line that begins something new, so it ends the card body above it even without a blank line.
        A bare "> " line stays inside a card: there it is a quote."""
        if not line or line.startswith(('//', '@', '==', 'title ')):
            return True
        if line.startswith('?') or CARD.fullmatch(line):
            return True
        m = SPEECH.fullmatch(line)
        return bool(m and (m.group('who') or m.group('src')))

    def block():
        nonlocal i
        body = []
        while i + 1 < len(lines) and not starts_statement(lines[i + 1].strip()):
            i += 1
            body.append(lines[i].strip())
        return body

    def paragraphs_and_buttons(body):
        paras, buttons = [], []
        for b in body:
            m = BUTTON.fullmatch(b)
            if m:
                buttons.append({'t': m.group('t'), 'go': m.group('go')})
            else:
                paras.append(b)
        return paras, buttons

    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith('//'):
            i += 1
            continue
        if line.startswith('title '):
            parts = [p.strip() for p in line[6:].split('|')]
            # name | years | title music | cover (chapters/<cover>.png, the chapter shelf) | theme (plays on the shelf)
            state['title'] = {'name': parts[0], 'span': parts[1] if len(parts) > 1 else '', 'music': parts[2] if len(parts) > 2 else '',
                              'cover': parts[3] if len(parts) > 3 else '', 'theme': parts[4] if len(parts) > 4 else ''}
        elif line.startswith('=='):
            parts = [p.strip() for p in line[2:].split('|')]
            if len(parts) != 3:
                fail('scene header needs "== id | menu label | place"')
            sid, label, place = parts
            state['p'] = place
            state['scenes'].append({'id': sid, 'label': label})
            pending['scene'] = sid
            pending.setdefault('fx', 'none')
            pending.setdefault('cg', '')
        elif line.startswith('@'):
            word, _, arg = line[1:].partition(' ')
            arg = arg.strip()
            if word == 'bg':
                pending['bg'] = arg
                pending.setdefault('cg', '')  # a new backdrop ends an illustration unless @cg is given with it
            elif word == 'cg':
                pending['cg'] = arg
            elif word == 'music':
                pending['music'] = arg
            elif word == 'fx':
                if arg not in ('dust', 'snow', 'rain', 'embers', 'smoke', 'none'):
                    fail('unknown effect @fx ' + arg)
                pending['fx'] = arg
            elif word in ('shake', 'flash'):
                pending.setdefault('once', []).append(word)
            elif word == 'sfx':  # a sound on the next line: sfx/<name>.mp3 (17.10)
                pending['sfx'] = arg
            elif word == 'on':
                pending['on'] = arg.split() if arg else []
                pending.setdefault('x', {})  # a new line-up drops old position overrides
            elif word == 'x':
                pending['x'] = {k: float(v) for k, v in (a.split('=') for a in arg.split())}
            elif word == 'e':
                pending['e'] = dict(a.split('=') for a in arg.split())
            elif word == 'id':
                pending['id'] = arg
            elif word == 'role':
                who, _, role = arg.partition('=')
                pending.setdefault('role', {})[who.strip()] = role.strip()
            else:
                fail('unknown directive @' + word)
        elif CHOICE.fullmatch(line) and line.startswith('?'):
            m = CHOICE.fullmatch(line)
            opts = [BUTTON.fullmatch(b) for b in block()]
            if not opts or not all(opts):
                fail('a choice needs lines "* text -> id"')
            step = {'k': 'choice', 'ask': m.group('ask')}
            _source(step, m.group('src'))
            step['opts'] = [{'t': o.group('t'), 'go': o.group('go')} for o in opts]
            take(step)
        elif CARD.fullmatch(line):
            m = CARD.fullmatch(line)
            kind = m.group(1)
            rest = [p.strip() for p in (m.group('rest') or '').split('|')] if m.group('rest') is not None else []
            body = block()
            step = {'k': kind}
            captions = [b[2:] for b in body if b.startswith('= ')]
            body = [b for b in body if not b.startswith('= ')]
            if captions:
                step['t'] = ' '.join(captions)
            _source(step, m.group('src'))
            if kind == 'paper':
                if len(rest) != 3:
                    fail('paper needs "paper N | masthead | date | headline"')
                step['mast'], step['date'], step['hd'] = rest
                step['bd'] = [b for b in body if not b.startswith('> ')]
                step['q'] = [b[2:] for b in body if b.startswith('> ')]
            elif kind == 'wire':
                if len(rest) != 2:
                    fail('wire needs "wire N | from | to"')
                step['from'], step['to'] = rest
                step['tape'] = body
            elif kind == 'letter':
                if len(rest) != 2:
                    fail('letter needs "letter N | heading | signature"')
                step['head'], step['sign'] = rest
                step['body'] = body
            elif kind == 'doc':
                if len(rest) != 1:
                    fail('doc needs "doc N | title"')
                step['title'] = rest[0]
                step['items'] = body
            elif kind == 'credits':
                if len(rest) not in (1, 2):
                    fail('credits needs "credits | heading | caption on the curtain before it"')
                step['title'] = rest[0]
                if len(rest) == 2:
                    step['end'] = rest[1]  # shown on the curtain between the last line and the roll («Конец пролога»)
                rows = []
                for b in body:
                    if b.startswith('# '):
                        rows.append({'h': b[2:].strip()})
                    elif '|' in b:
                        parts = [x.strip() for x in b.split('|')]
                        if len(parts) > 3:
                            fail('a credits row is "who | Полное имя | player" or "текст | имя"')
                        row = {'l': parts[0], 'r': parts[-1]}
                        if len(parts) == 3:
                            row['f'] = parts[1]  # the full name, shown beside the photo
                        rows.append(row)
                    else:
                        rows.append({'t': b})
                if not rows:
                    fail('credits needs at least one line "who | player"')
                step['rows'] = rows
            elif kind == 'fork':
                step['stamp'] = rest[0] if rest else 'Так в игре не было'
                paras, step['buttons'] = paragraphs_and_buttons(body)
                step['paras'] = [p for p in paras if not p.startswith('> ')]
                step['note'] = ' '.join(p[2:] for p in paras if p.startswith('> '))
            else:  # end
                if len(rest) != 2:
                    fail('end needs "end | date | title"')
                step['date'], step['title'] = rest
                paras, step['buttons'] = paragraphs_and_buttons(body)
                step['paras'] = [p for p in paras if not p.startswith('> ')]
                step['note'] = ' '.join(p[2:] for p in paras if p.startswith('> '))
            if kind in ('fork', 'end') and not step['buttons']:
                fail(kind + ' needs at least one button "* text -> id"')
            take(step)
        else:
            m = SPEECH.fullmatch(line)
            if not m:
                fail('cannot read this line: ' + line[:60])
            step = {'k': 'say', 't': m.group('t')}
            _source(step, m.group('src'))
            if m.group('who'):
                step['w'] = m.group('who')
                if m.group('emo'):
                    step['e'] = {step['w']: m.group('emo')}
            take(step)
        i += 1
    if pending.get('scene') or pending.get('id'):
        fail('a scene or @id at the end of the file has no line after it')
    return steps, state


def load_chapter(folder):
    """All steps of a chapter folder, the chapter title and its scene list."""
    steps, state = [], None
    for name in sorted(os.listdir(folder)):
        if name.endswith('.txt'):
            with open(os.path.join(folder, name), encoding='utf-8') as f:
                part, state = parse(f.read(), name, state)
            steps.extend(part)
    return steps, state


def load_bios(folder):
    """Character biographies (17.10): scenario/bio/<who>.txt — the first line is the full name, then sections
    "== Название главы" with paragraphs (a blank line between them, "//" comments; a line "# …" is a subheading inside the
    section and is kept as a paragraph starting with "# "). { who: {full, sections: [{ch, paras}]} }.
    A character with a file gets a card in the game's «Персонажи»; the others (one-scene people, voices) do not."""
    bios = {}
    if not os.path.isdir(folder):
        return bios
    for name in sorted(os.listdir(folder)):
        if not name.endswith('.txt'):
            continue
        who = name[:-4]
        with open(os.path.join(folder, name), encoding='utf-8') as f:
            lines = [l.rstrip() for l in f.read().splitlines()]
        full, sections, para = '', [], []
        def flush():
            if para and sections:
                sections[-1]['paras'].append(' '.join(para))
            para.clear()
        def sub(t):  # a subheading stands alone
            flush()
            sections[-1]['paras'].append(t)
        for line in lines:
            t = line.strip()
            if t.startswith('//'):
                continue
            if not full:
                if t:
                    full = t
                continue
            if t.startswith('=='):
                flush()
                sections.append({'ch': t[2:].strip(), 'paras': []})
            elif not t:
                flush()
            else:
                if not sections:
                    raise ScenarioError('%s: text before the first "== Глава" section' % name)
                if t.startswith('# '):
                    sub(t)
                else:
                    para.append(t)
        flush()
        if not full:
            raise ScenarioError('%s: the first line must be the full name' % name)
        bios[who] = {'full': full, 'sections': sections}
    return bios


def check(steps, characters, backgrounds):
    """Problems a reader would hit: unknown speakers, broken jumps. Returns a list of strings."""
    problems = []
    ids = {s['id'] for s in steps if s.get('id')}
    for n, s in enumerate(steps):
        if s.get('w') and s['w'] not in characters:
            problems.append('step %d: unknown speaker %r' % (n, s['w']))
        for who in (s.get('on') or []) + list((s.get('e') or {}).keys()):
            if who not in characters:
                problems.append('step %d: unknown character %r' % (n, who))
        for b in s.get('opts', []) + s.get('buttons', []):
            if b['go'] not in ids and b['go'] not in ('restart', 'next', 'title'):
                problems.append('step %d: jump to unknown id %r' % (n, b['go']))
        for r in s.get('rows', []):  # a credits row whose left side looks like a character id must name one
            if r.get('l') and re.fullmatch(r'[a-z_]+', r['l']) and r['l'] not in characters:
                problems.append('step %d: credits for unknown character %r' % (n, r['l']))
    return problems
