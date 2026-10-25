"""Every \caption in the files main.tex actually inputs: its longest plain-text run and its last
long run must appear in the PDF text (catches captions silently clipped, e.g. by wrapfigure)."""
import re, sys, unicodedata
txt = open(sys.argv[1], encoding='utf-8').read()

def norm(s):
    s = unicodedata.normalize('NFKC', s)
    for a, b in [('---', '-'), ('--', '-'), ('\u2013', '-'), ('\u2014', '-'), ('\u2019', "'"), ('\u2018', "'"),
                 ('``', '"'), ("''", '"'), ('\u201c', '"'), ('\u201d', '"'), ('\u2212', '-'), ('`', "'")]:
        s = s.replace(a, b)
    s = re.sub(r'-\s*\n\s*', '', s).replace('-', '')
    return re.sub(r'\s+', ' ', s).lower()

T = norm(txt)
files = []
def walk(f):
    try:
        s = open(f, encoding='utf-8').read()
    except FileNotFoundError:
        return
    files.append(f)
    for m in re.finditer(r'^[^%\n]*?\\input\{([^}]+)\}', s, re.M):
        g = m.group(1); walk(g if g.endswith('.tex') else g + '.tex')
walk('main.tex')

def captions(s):
    out, i = [], 0
    while (j := s.find('\\caption{', i)) >= 0:
        k, d = j + 9, 1
        while d and k < len(s):
            if s[k] == '\\': k += 2; continue
            d += {'{': 1, '}': -1}.get(s[k], 0); k += 1
        out.append(s[j + 9:k - 1]); i = k
    return out

n = bad = 0
for f in files:
    for c in captions(re.sub(r'(?<!\\)%.*', '', open(f, encoding='utf-8').read())):
        n += 1
        parts = [p.strip(' .,;:()') for p in re.split(r'\\[a-zA-Z]+\*?|\\.|[{}$~]|\d+(?:\.\d+)?', c)]
        long = [p for p in parts if len(p) > 25]
        for fr in {max(parts, key=len), long[-1] if long else max(parts, key=len)}:
            if norm(fr) not in T:
                bad += 1; print('MISSING', f, '|', fr[:100])
print(f'captions={n} files={len(files)} missing={bad}')
