#!/usr/bin/env python3
"""status_query.py — CLI renderer + editor for Obsidian Bases over _status/ frontmatter.

WHY (P79 doc architecture):
  docs/checkpoints/_status/**/*.md frontmatter 是 single-source 半静态数据层
  (cells / tasks / issues / codex / paper-sections). docs/*.base 的 5 个文件定义
  视图 (filter + 列 order + formula 图标). 但 Bases 只在 Obsidian 端渲染 —— 从 CLI
  之前只能看到裸的 `![[tasks.base#NOW]]` embed 指针 (空壳). 本工具在终端渲染**同一份**
  .base 视图 (与 Obsidian 字节等价), 并支持 `set` 改 frontmatter 字段, 这样我们不再
  维护一张平行的手写表。single source = .base + frontmatter, 两端一致。

它实现了一个够用的 Bases 表达式子集 (覆盖本仓 5 个 .base 的全部语法):
  - filters:  field == "x" | field != "x" | file.inFolder("p") | <field truthy>, and/or/not
  - formulas: if(cond, then, else) 嵌套 · == != >= <= > < · + (字符串拼接/数字加)
              · / * - · "字面量" · 字段引用 · null · x.toString() · (分组)
  - views:    table / cards · 每视图 filter · order (field|file.name|formula.X)
              · groupBy {property,direction} · summaries {field: Sum}

USAGE:
  status_query.py                          # 列出每个 .base + 其视图名 + note 数
  status_query.py tasks                    # 渲染 tasks.base 全部视图
  status_query.py tasks#NOW                # 渲染单视图 (名字子串匹配, 大小写无关)
  status_query.py cells --view "Active"    # 同上, 显式 --view
  status_query.py --json issues#Active     # 机器可读 (每行一个 dict)
  status_query.py set cell_b0_cls_vision status=active blocker="GPU contention"

解析: <base> 匹配 docs/<base>.base (stem). '#' 之后或 --view = 视图名的大小写无关子串。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
VAULT = REPO_ROOT / "docs"                       # .base 文件所在 (Obsidian vault root)
STATUS_DIR = VAULT / "checkpoints" / "_status"   # frontmatter notes 根


# ───────────────────────── frontmatter ─────────────────────────
def parse_frontmatter(text: str):
    """返回 (frontmatter_dict, body_str). 无 YAML front block 时 dict 为空。"""
    if not text.startswith("---"):
        return {}, text
    m = re.match(r"^---\n(.*?)\n---\n?(.*)$", text, re.DOTALL)
    if not m:
        return {}, text
    try:
        fm = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        fm = {}
    return (fm if isinstance(fm, dict) else {}), m.group(2)


class Note:
    """一个 _status note: frontmatter + 文件名 (Bases 的 file.name = stem, 无扩展名)。"""

    def __init__(self, path: Path):
        self.path = path
        self.name = path.stem
        self.fm, self.body = parse_frontmatter(path.read_text(encoding="utf-8"))

    def rel(self) -> str:
        return str(self.path.relative_to(VAULT)).replace("\\", "/")


def scan_notes():
    notes = []
    if not STATUS_DIR.exists():
        return notes
    for p in sorted(STATUS_DIR.rglob("*.md")):
        if p.suffix != ".md":          # 跳过 *.md.lock 等 Obsidian 锁文件
            continue
        notes.append(Note(p))
    return notes


# ───────────────────── 表达式求值器 (Bases 子集) ─────────────────────
# tokenizer: 数字 / 字符串 / 运算符 / 标点 / 标识符
_TOKEN_RE = re.compile(
    r"""(?P<num>\d+(?:\.\d+)?)
      | (?P<str>"(?:[^"\\]|\\.)*")
      | (?P<op>==|!=|>=|<=|>|<|\+|\-|\*|/)
      | (?P<punct>[(),.])
      | (?P<ident>[A-Za-z_][A-Za-z0-9_]*)
    """,
    re.VERBOSE,
)


class Tok:
    __slots__ = ("kind", "val")

    def __init__(self, kind, val):
        self.kind, self.val = kind, val

    def __repr__(self):
        return f"{self.kind}:{self.val!r}"


def _unquote(s: str) -> str:
    return s[1:-1].replace('\\"', '"').replace("\\\\", "\\")


def tokenize(expr: str):
    toks, pos, n = [], 0, len(expr)
    while pos < n:
        if expr[pos].isspace():
            pos += 1
            continue
        m = _TOKEN_RE.match(expr, pos)
        if not m:
            raise ValueError(f"无法 tokenize @ {pos}: {expr[pos:pos+20]!r}")
        pos = m.end()
        g = m.lastgroup
        if g == "num":
            toks.append(Tok("num", float(m.group())))
        elif g == "str":
            toks.append(Tok("str", _unquote(m.group())))
        elif g == "op":
            toks.append(Tok("op", m.group()))
        elif g == "punct":
            toks.append(Tok("punct", m.group()))
        else:
            toks.append(Tok("ident", m.group()))
    return toks


class Parser:
    """递归下降: compare(最低) → add/mul(左结合, 同级) → postfix(.method(args)) → primary.

    注: 只两级优先级 (比较 < 算术). 本仓 .base 表达式要么纯比较、要么纯拼接、
    要么用括号隔开除法, 故无需更细粒度优先级。"""

    def __init__(self, toks):
        self.toks, self.i = toks, 0

    def _peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else None

    def _next(self):
        t = self.toks[self.i]
        self.i += 1
        return t

    def _expect(self, kind, val=None):
        t = self._next()
        if t.kind != kind or (val is not None and t.val != val):
            raise ValueError(f"期望 {kind} {val}, 得到 {t}")
        return t

    def parse(self):
        node = self._compare()
        if self._peek() is not None:
            raise ValueError(f"多余 token: {self.toks[self.i:]}")
        return node

    def _compare(self):
        left = self._arith()
        t = self._peek()
        if t and t.kind == "op" and t.val in ("==", "!=", ">=", "<=", ">", "<"):
            self._next()
            return ("cmp", t.val, left, self._arith())
        return left

    def _arith(self):
        left = self._postfix()
        while True:
            t = self._peek()
            if t and t.kind == "op" and t.val in ("+", "-", "*", "/"):
                self._next()
                left = ("bin", t.val, left, self._postfix())
            else:
                return left

    def _postfix(self):
        node = self._primary()
        while self._peek() and self._peek().kind == "punct" and self._peek().val == ".":
            self._next()
            meth = self._expect("ident").val
            self._expect("punct", "(")
            args = []
            if not (self._peek() and self._peek().val == ")"):
                args.append(self._compare())
                while self._peek() and self._peek().val == ",":
                    self._next()
                    args.append(self._compare())
            self._expect("punct", ")")
            node = ("method", meth, node, args)
        return node

    def _primary(self):
        t = self._next()
        if t.kind == "num":
            return ("num", t.val)
        if t.kind == "str":
            return ("str", t.val)
        if t.kind == "punct" and t.val == "(":
            node = self._compare()
            self._expect("punct", ")")
            return node
        if t.kind == "ident":
            if t.val == "null":
                return ("null",)
            if t.val == "if":
                self._expect("punct", "(")
                cond = self._compare()
                self._expect("punct", ",")
                a = self._compare()
                self._expect("punct", ",")
                b = self._compare()
                self._expect("punct", ")")
                return ("if", cond, a, b)
            return ("field", t.val)
        raise ValueError(f"意外 token {t}")


# ── 值语义 (JS-like) ──
def _num(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return None
    return None


def _s(v):
    """stringify, JS toString 风格: None→'' / bool→true|false / 整数浮点去 .0。"""
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def _eq(l, r):
    if l is None or r is None:
        return l is None and r is None
    ln, rn = _num(l), _num(r)
    if ln is not None and rn is not None:
        return ln == rn
    return str(l) == str(r)


def truthy(v):
    if v is None:
        return False
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v != 0
    if isinstance(v, str):
        return v.strip() != ""
    return bool(v)


def _compare_op(op, l, r):
    if op == "==":
        return _eq(l, r)
    if op == "!=":
        return not _eq(l, r)
    ln, rn = _num(l), _num(r)
    if ln is None or rn is None:
        ls, rs = str(l), str(r)
        return {">": ls > rs, "<": ls < rs, ">=": ls >= rs, "<=": ls <= rs}[op]
    return {">": ln > rn, "<": ln < rn, ">=": ln >= rn, "<=": ln <= rn}[op]


def _bin_op(op, l, r):
    if op == "+":
        if isinstance(l, str) or isinstance(r, str):
            return _s(l) + _s(r)
        ln, rn = _num(l), _num(r)
        return (ln + rn) if (ln is not None and rn is not None) else _s(l) + _s(r)
    ln, rn = _num(l), _num(r)
    if ln is None or rn is None:
        return None
    if op == "-":
        return ln - rn
    if op == "*":
        return ln * rn
    if op == "/":
        return ln / rn if rn else None
    return None


def _eval(node, note: Note):
    k = node[0]
    if k == "num":
        return node[1]
    if k == "str":
        return node[1]
    if k == "null":
        return None
    if k == "field":
        f = node[1]
        return note if f == "file" else note.fm.get(f)
    if k == "cmp":
        return _compare_op(node[1], _eval(node[2], note), _eval(node[3], note))
    if k == "bin":
        return _bin_op(node[1], _eval(node[2], note), _eval(node[3], note))
    if k == "if":
        return _eval(node[2], note) if truthy(_eval(node[1], note)) else _eval(node[3], note)
    if k == "method":
        meth, base, args = node[1], node[2], node[3]
        bval = _eval(base, note)
        eargs = [_eval(a, note) for a in args]
        if meth == "toString":
            return _s(bval)
        if meth == "inFolder":
            return (eargs[0] in note.rel()) if eargs else False
        if meth == "name":
            return note.name
        return None
    raise ValueError(f"未知 AST 节点 {node}")


_PARSE_CACHE: dict = {}


def eval_expr(expr: str, note: Note):
    expr = expr.strip()
    ast = _PARSE_CACHE.get(expr)
    if ast is None:
        ast = Parser(tokenize(expr)).parse()
        _PARSE_CACHE[expr] = ast
    return _eval(ast, note)


def eval_filter(spec, note: Note) -> bool:
    """filter spec: None / 字符串表达式 / {and|or|not: ...} / list (隐式 and)。"""
    if spec is None:
        return True
    if isinstance(spec, str):
        return truthy(eval_expr(spec, note))
    if isinstance(spec, dict):
        if "and" in spec:
            return all(eval_filter(s, note) for s in spec["and"])
        if "or" in spec:
            return any(eval_filter(s, note) for s in spec["or"])
        if "not" in spec:
            return not eval_filter(spec["not"], note)
        return True
    if isinstance(spec, list):
        return all(eval_filter(s, note) for s in spec)
    return True


# ───────────────────────── 列解析 ─────────────────────────
def col_header(col: str, props: dict) -> str:
    p = props.get(col)
    if isinstance(p, dict) and "displayName" in p:
        return p["displayName"]
    if col == "file.name":
        return "Name"
    if col.startswith("formula."):
        return col[len("formula."):]
    return col


def _compact(v):
    if isinstance(v, list):
        return f"[{len(v)} items]"
    if isinstance(v, dict):
        return "{…}"
    return str(v)


def col_value(col: str, note: Note, formulas: dict):
    if col == "file.name":
        return note.name
    if col.startswith("formula."):
        expr = formulas.get(col[len("formula."):])
        if expr is None:
            return ""
        try:
            return _s(eval_expr(expr, note))
        except Exception as e:  # formula 错误不该杀整张表
            return f"<err:{e}>"
    val = note.fm.get(col)
    return _s(val) if not isinstance(val, (list, dict)) else _compact(val)


# ───────────────────────── 终端宽度 / 渲染 ─────────────────────────
_EMOJI_RANGES = (
    (0x1F300, 0x1FAFF), (0x2600, 0x27BF), (0x2B00, 0x2BFF),
    (0x2300, 0x23FF), (0x1F1E6, 0x1F1FF),
)


def _cw(ch: str) -> int:
    o = ord(ch)
    if o == 0xFE0F or unicodedata.combining(ch):
        return 0
    if unicodedata.east_asian_width(ch) in ("W", "F"):
        return 2
    for a, b in _EMOJI_RANGES:
        if a <= o <= b:
            return 2
    return 1


def display_width(s) -> int:
    return sum(_cw(c) for c in str(s))


def _pad(s, width: int) -> str:
    return str(s) + " " * max(0, width - display_width(s))


def _clip(s, maxw: int) -> str:
    s = str(s)
    if maxw <= 0 or display_width(s) <= maxw:
        return s
    out, w = "", 0
    for ch in s:
        cw = _cw(ch)
        if w + cw > maxw - 1:
            break
        out += ch
        w += cw
    return out + "…"


def _render_rows(headers, rows):
    """对齐打印 (考虑中文/emoji 宽度). rows = list[list[str]]."""
    ncol = len(headers)
    widths = [display_width(h) for h in headers]
    for r in rows:
        for i in range(ncol):
            widths[i] = max(widths[i], display_width(r[i]))
    out = []
    out.append("  ".join(_pad(headers[i], widths[i]) for i in range(ncol)).rstrip())
    out.append("  ".join("─" * widths[i] for i in range(ncol)))
    for r in rows:
        out.append("  ".join(_pad(r[i], widths[i]) for i in range(ncol)).rstrip())
    return "\n".join(out)


def render_view(view: dict, base: dict, base_notes, maxw: int) -> str:
    props = base.get("properties", {}) or {}
    formulas = base.get("formulas", {}) or {}
    cols = view.get("order") or _auto_cols(base_notes)
    rows_notes = [n for n in base_notes if eval_filter(view.get("filters"), n)]

    headers = [col_header(c, props) for c in cols]
    lines = [f"  ┌─ #{view.get('name','(view)')}  ({len(rows_notes)})"]

    gb = view.get("groupBy")
    if gb and gb.get("property"):
        prop = gb["property"]
        groups: dict = {}
        for n in rows_notes:
            groups.setdefault(_s(n.fm.get(prop)), []).append(n)
        keys = sorted(groups, reverse=(gb.get("direction", "ASC").upper() == "DESC"))
        for key in keys:
            disp = key if key != "" else "(none)"
            lines.append(f"\n  ▸ {prop} = {disp}  ({len(groups[key])})")
            body = _render_rows(
                headers, [[_clip(col_value(c, n, formulas), maxw) for c in cols] for n in groups[key]]
            )
            lines.append(_indent(body))
    else:
        body = _render_rows(
            headers, [[_clip(col_value(c, n, formulas), maxw) for c in cols] for n in rows_notes]
        )
        lines.append(_indent(body))

    for field, kind in (view.get("summaries") or {}).items():
        if str(kind).lower() == "sum":
            total = sum((_num(n.fm.get(field)) or 0.0) for n in rows_notes)
            lines.append(f"  Σ {field} = {_s(total)}")
    return "\n".join(lines)


def _indent(block: str, pad="    ") -> str:
    return "\n".join(pad + ln for ln in block.split("\n"))


def _auto_cols(notes):
    seen = []
    for n in notes:
        for k in n.fm:
            if k != "type" and k not in seen:
                seen.append(k)
    return ["file.name"] + seen[:6]


# ───────────────────────── JSON 输出 ─────────────────────────
def view_rows_json(view, base, base_notes):
    formulas = base.get("formulas", {}) or {}
    cols = view.get("order") or _auto_cols(base_notes)
    rows_notes = [n for n in base_notes if eval_filter(view.get("filters"), n)]
    out = []
    for n in rows_notes:
        row = {"file.name": n.name}
        for c in cols:
            row[c] = col_value(c, n, formulas)
        out.append(row)
    return out


# ───────────────────────── base 加载 / 选择 ─────────────────────────
def load_base(stem: str):
    bf = VAULT / f"{stem}.base"
    if not bf.exists():
        avail = ", ".join(p.stem for p in sorted(VAULT.glob("*.base")))
        sys.exit(f"找不到 base '{stem}'. 可用: {avail}")
    return yaml.safe_load(bf.read_text(encoding="utf-8")) or {}


def pick_views(base: dict, selector: str):
    views = base.get("views", []) or []
    if not selector:
        return views
    sel = selector.lower()
    matched = [v for v in views if sel in str(v.get("name", "")).lower()]
    if not matched:
        names = " / ".join(v.get("name", "?") for v in views)
        sys.exit(f"视图 '{selector}' 无匹配. 可选: {names}")
    return matched


# ───────────────────────── 命令 ─────────────────────────
def cmd_list():
    notes = scan_notes()
    print(f"_status notes: {len(notes)}  |  vault: {VAULT}\n")
    for bf in sorted(VAULT.glob("*.base")):
        base = yaml.safe_load(bf.read_text(encoding="utf-8")) or {}
        n = sum(1 for x in notes if eval_filter(base.get("filters"), x))
        print(f"● {bf.stem:<8} ({n:>2} notes)")
        for v in base.get("views", []) or []:
            cnt = sum(
                1 for x in notes
                if eval_filter(base.get("filters"), x) and eval_filter(v.get("filters"), x)
            )
            print(f"    #{v.get('name','?'):<28} {cnt:>2}")
    print("\n用法: status_query.py <base>[#视图子串]   |   set <note> field=value")


def cmd_render(target: str, view_sel: str, as_json: bool, maxw: int):
    stem, _, inline_sel = target.partition("#")
    selector = view_sel or inline_sel
    base = load_base(stem)
    notes = scan_notes()
    base_notes = [n for n in notes if eval_filter(base.get("filters"), n)]
    views = pick_views(base, selector)

    if as_json:
        payload = {v.get("name"): view_rows_json(v, base, base_notes) for v in views}
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    print(f"━━ {stem}.base  ({len(base_notes)} notes) ━━")
    for v in views:
        print()
        print(render_view(v, base, base_notes, maxw))


def _coerce(raw: str):
    """set 的值类型推断: 空→'' / 整数→int / 浮点→float / true|false→bool / 否则 str。"""
    s = raw.strip()
    if s == "":
        return ""
    low = s.lower()
    if low in ("true", "false"):
        return low == "true"
    if re.fullmatch(r"-?\d+", s):
        return int(s)
    if re.fullmatch(r"-?\d+\.\d+", s):
        return float(s)
    return s


def _find_note(name: str) -> Path:
    cands = []
    for p in STATUS_DIR.rglob("*.md"):
        if p.suffix != ".md":
            continue
        if p.stem == name:
            return p
        if name in p.stem:
            cands.append(p)
    if len(cands) == 1:
        return cands[0]
    if not cands:
        sys.exit(f"找不到 note '{name}' (在 {STATUS_DIR})")
    sys.exit(f"note '{name}' 不唯一: {', '.join(p.stem for p in cands)}")


def set_field(path: Path, field: str, raw: str):
    """行级替换 frontmatter 顶层字段 (不 round-trip 整个 YAML, 保留 history 等复杂字段格式)。"""
    text = path.read_text(encoding="utf-8")
    lines = text.split("\n")
    if not lines or lines[0].strip() != "---":
        sys.exit(f"{path.name}: 无 frontmatter 起始 '---'")
    try:
        end = lines.index("---", 1)
    except ValueError:
        sys.exit(f"{path.name}: 无 frontmatter 结束 '---'")
    val = _coerce(raw)
    new_line = yaml.safe_dump({field: val}, allow_unicode=True, default_flow_style=False).strip()
    pat = re.compile(rf"^{re.escape(field)}\s*:")
    for i in range(1, end):
        if pat.match(lines[i]):
            old = lines[i]
            lines[i] = new_line
            path.write_text("\n".join(lines), encoding="utf-8")
            return f"  {field}: {old.split(':',1)[1].strip()}  →  {new_line.split(':',1)[1].strip()}"
    lines.insert(end, new_line)          # 字段不存在 → 插入到结束 fence 前
    path.write_text("\n".join(lines), encoding="utf-8")
    return f"  + {new_line}  (新增)"


def cmd_set(rest):
    if len(rest) < 2:
        sys.exit("用法: status_query.py set <note> field=value [field2=value2 ...]")
    name, assigns = rest[0], rest[1:]
    path = _find_note(name)
    print(f"✎ {path.relative_to(VAULT)}")
    for a in assigns:
        if "=" not in a:
            sys.exit(f"赋值需 field=value, 得到: {a!r}")
        field, _, val = a.partition("=")
        print(set_field(path, field.strip(), val))


# ───────────────────────── main ─────────────────────────
def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "set":
        return cmd_set(sys.argv[2:])
    ap = argparse.ArgumentParser(description="CLI renderer for Obsidian Bases over _status/ frontmatter")
    ap.add_argument("target", nargs="?", help="<base>[#视图子串], 省略=列出全部")
    ap.add_argument("--view", default="", help="视图名子串 (等价 #后缀)")
    ap.add_argument("--json", action="store_true", help="机器可读输出")
    ap.add_argument("--width", type=int, default=50, help="单列最大显示宽度 (默认 50)")
    args = ap.parse_args()
    if not args.target:
        return cmd_list()
    cmd_render(args.target, args.view, args.json, args.width)


if __name__ == "__main__":
    main()
