import json, sys, os, collections
src_root, out_dir = sys.argv[1], sys.argv[2]
def flat(t):
    if isinstance(t, str): return t
    return ''.join(x if isinstance(x, str) else x.get('text','') for x in t)
for d in sorted(os.listdir(src_root)):
    if d == 'rp-archive': continue
    p = os.path.join(src_root, d)
    if not os.path.isdir(p): continue
    for f in os.listdir(p):
        if not f.endswith('.json'): continue
        data = json.load(open(os.path.join(p, f), encoding='utf-8'))
        msgs = data.get('messages', [])
        authors = collections.Counter()
        chars = 0; n = 0; first = last = None
        name = 'ch_%02d' % len(os.listdir(out_dir))
        with open(os.path.join(out_dir, name + '.txt'), 'w', encoding='utf-8') as o:
            o.write('# ' + d + ' | ' + data.get('name','') + '\n')
            for m in msgs:
                if m.get('type') != 'message': continue
                txt = flat(m.get('text',''))
                a = m.get('from') or m.get('author') or '?'
                if not txt.strip() and not m.get('media_type') and not m.get('photo'): continue
                n += 1; chars += len(txt); authors[a] += 1
                dt = m.get('date','')[:16]
                first = first or dt; last = dt
                media = ' [%s]' % (m.get('media_type') or ('photo' if m.get('photo') else '')) if (m.get('media_type') or m.get('photo')) else ''
                o.write('\n=== #%s %s | %s%s\n%s\n' % (m.get('id'), dt, a, media, txt))
        print(name, repr(d.encode('ascii','backslashreplace'))[:0], 'msgs=%d chars=%d %s..%s authors=%d' % (n, chars, first, last, len(authors)))
        print('   top:', [ (k.encode('unicode_escape').decode()[:0] or k, v) for k, v in authors.most_common(8)][:0] or '', file=sys.stdout)
        with open(os.path.join(out_dir, name + '.meta'), 'w', encoding='utf-8') as o:
            o.write(d + '\n' + json.dumps(authors.most_common(25), ensure_ascii=False))
