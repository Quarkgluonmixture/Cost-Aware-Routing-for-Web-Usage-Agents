#!/usr/bin/env bash
# overleaf_sync.sh — md 真源 → convert.sh → Overleaf git 单向同步
#
# 工作流 (advisor 2026-07-16 决定: Overleaf git + 学长 collaborator 看 draft):
#   - 唯一真源仍是 docs/checkpoints/paper_drafts/aaai27/aaai27_main.md
#   - 本脚本重跑 convert.sh, 把编译产物集拷进 Overleaf git clone, commit+push
#   - 学长在 Overleaf 网页端看/评论; 他的直接编辑不自动回流 —— 内容冻结前
#     一律由我们改 md 再 sync (防双源漂移); 冻结后再切换 Overleaf 为终稿层
#
# 首次配置:
#   1. user 在 Overleaf 新建项目 (上传 aaai27_overleaf_*.zip), Menu → Git 拿 URL
#   2. git clone https://git.overleaf.com/<PROJECT_ID> ~/overleaf-aaai27
#      (密码 = Overleaf Account Settings → Git authentication token)
#   3. export OVERLEAF_GIT_DIR=~/overleaf-aaai27 (或写入 scripts/vwa_env_remote.sh)
#
# 用法: bash scripts/maintenance/overleaf_sync.sh ["commit message"]

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
LATEX_DIR="$REPO/docs/checkpoints/paper_drafts/aaai27/latex"
BUILD="$LATEX_DIR/build"
OL="${OVERLEAF_GIT_DIR:-}"
MSG="${1:-sync: aaai27_main.md $(cd "$REPO" && git rev-parse --short HEAD)}"

if [[ -z "$OL" || ! -d "$OL/.git" ]]; then
  echo "✗ OVERLEAF_GIT_DIR 未设置或不是 git clone (见脚本头部首次配置)" >&2
  exit 1
fi

echo "[1/3] convert.sh 重新生成..."
(cd "$LATEX_DIR" && bash convert.sh > /tmp/overleaf_sync_convert.log 2>&1) \
  || { echo "✗ convert.sh 失败, 见 /tmp/overleaf_sync_convert.log"; exit 2; }
grep -E "PDF:|Reference start page" /tmp/overleaf_sync_convert.log || true

echo "[2/3] 拷贝产物集 → $OL"
mkdir -p "$OL/figures"
cp "$BUILD"/main.tex "$BUILD"/title_generated.tex "$BUILD"/abstract_generated.tex \
   "$BUILD"/body_generated.tex "$BUILD"/aaai2027.sty "$BUILD"/aaai2027.bst \
   "$BUILD"/paper.bib "$OL/"
cp "$BUILD"/figures/*.pdf "$OL/figures/" 2>/dev/null || true

echo "[3/3] commit + push"
cd "$OL"
git add -A
if git diff --cached --quiet; then
  echo "· 无变化, 跳过 push"
else
  git commit -q -m "$MSG"
  git push -q origin master 2>/dev/null || git push -q origin main
  echo "✓ pushed: $MSG"
fi
