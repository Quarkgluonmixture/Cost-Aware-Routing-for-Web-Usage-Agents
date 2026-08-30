#!/usr/bin/env bash
# overleaf_thesis_sync.sh — final_dissertation/tex (真源) → Overleaf git 单向同步
#
# 工作流 (2026-08-11):
#   - 真源 = 本仓库 final_dissertation/tex/。改内容一律改这里, 然后跑本脚本。
#   - Overleaf 项目 6a8f68ace4443ea9751d6201 = 渲染 / 导师评审层。
#     2026-08-27 换项目: 稿件迁到 UCL PhD Thesis Template, 新项目就是从该模板
#     开的那一个。旧项目 6a7a7331d2e6523a360245d4 (自建 report 版式) 停用,
#     不要再往那边推 —— 两边版式不同, 推错会让导师看到旧排版。
#   - ⚠️ 单向。Overleaf 网页端的直接编辑**不会**回流; 内容冻结前一律由我们改
#     tex 再 sync (防双源漂移 —— 这正是 overleaf_sync.sh 对 REALM 稿定下的规矩)。
#     冻稿后若要切换成 Overleaf 为终稿层, 先把那边 pull 回来再改规矩。
#
# 前置: OVERLEAF_THESIS_DIR 指向已 clone 的 Overleaf 仓库 (默认 ~/overleaf-thesis)。
#       clone 时密码 = Overleaf Account Settings → Git authentication token。
#
# 用法:
#   bash scripts/maintenance/overleaf_thesis_sync.sh              # 同步 + commit + push
#   DRY_RUN=1 bash scripts/maintenance/overleaf_thesis_sync.sh    # 只看会传什么
#   NO_PUSH=1 bash scripts/maintenance/overleaf_thesis_sync.sh    # 传+commit, 不 push
set -euo pipefail

# Dry run is DRY_RUN=1 in the environment; this script takes no positional
# arguments. Before this guard an unrecognised argument was silently ignored,
# so `--dry-run` (the shape every other tool uses) ran a REAL sync to Overleaf
# while the caller believed nothing had moved. Exit-code 0, wrong action.
if [ "$#" -gt 0 ]; then
  echo "ERROR: this script takes no arguments (got: $*)." >&2
  echo "       Dry run is:  DRY_RUN=1 bash $0" >&2
  exit 2
fi


REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC="$REPO_ROOT/final_dissertation/tex"
DEST="${OVERLEAF_THESIS_DIR:-$HOME/overleaf-thesis-ucl}"

[ -d "$SRC" ]        || { echo "✗ 源目录不存在: $SRC"; exit 1; }
[ -d "$DEST/.git" ]  || { echo "✗ Overleaf clone 不存在: $DEST
  先跑: git clone https://git.overleaf.com/6a8f68ace4443ea9751d6201 $DEST"; exit 1; }

# 本地先编译一次 —— 不让编不过的稿子进 Overleaf。Overleaf 那边编译失败时
# 导师看到的是错误页而不是论文, 比晚同步几分钟糟得多。
if command -v latexmk >/dev/null 2>&1; then
  echo "→ 本地预编译..."
  ( cd "$SRC" && latexmk -pdf -interaction=nonstopmode main.tex >/tmp/overleaf_thesis_build.log 2>&1 ) \
    || { echo "✗ 本地编译失败, 拒绝同步。看 /tmp/overleaf_thesis_build.log"; exit 1; }
  UNDEF=$(grep -cE "LaTeX Warning: (Citation|Reference).*undefined" "$SRC/main.log" || true)
  [ "$UNDEF" = "0" ] || echo "⚠ 仍有 $UNDEF 处 undefined ref/citation (不阻断)"
  echo "  ✓ 编译通过 ($(pdfinfo "$SRC/main.pdf" 2>/dev/null | awk '/Pages/{print $2}') 页)"
fi

RSYNC_ARGS=(-a --delete
  --exclude '.git/' --exclude '*.aux' --exclude '*.log' --exclude '*.out'
  --exclude '*.toc' --exclude '*.lof' --exclude '*.lot' --exclude '*.fls'
  --exclude '*.fdb_latexmk' --exclude '*.bbl' --exclude '*.blg'
  --exclude '*.synctex.gz' --exclude 'main.pdf')

if [ -n "${DRY_RUN:-}" ]; then
  rsync "${RSYNC_ARGS[@]}" --dry-run -v "$SRC/" "$DEST/"
  echo "(DRY_RUN — 什么都没改)"
  exit 0
fi

rsync "${RSYNC_ARGS[@]}" "$SRC/" "$DEST/"

cd "$DEST"
if git diff --quiet && git diff --cached --quiet && [ -z "$(git status --porcelain)" ]; then
  echo "✓ Overleaf 已是最新, 无需 commit"
  exit 0
fi

git add -A
git commit -q -m "sync thesis from P79 repo ($(date -u +%Y-%m-%dT%H:%MZ))"
echo "✓ committed: $(git log -1 --oneline)"

if [ -n "${NO_PUSH:-}" ]; then
  echo "(NO_PUSH — 没 push)"
  exit 0
fi
git push -q origin HEAD 2>&1 | sed -E 's/olp_[A-Za-z0-9]+/olp_***/g'
echo "✓ pushed → Overleaf"
