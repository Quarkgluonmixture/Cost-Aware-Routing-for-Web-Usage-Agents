#!/usr/bin/env bash
# overleaf_sync.sh — md 真源 → convert.sh → Overleaf git 单向同步
#
# 工作流 (advisor 2026-07-16 决定: Overleaf git + 学长 collaborator 看 draft):
#   - 唯一真源是 docs/checkpoints/paper_drafts/<paper>/section*.md
#   - 本脚本重跑 convert.sh, 把编译产物集拷进 Overleaf git clone, commit+push
#   - 学长在 Overleaf 网页端看/评论; 他的直接编辑不自动回流 —— 内容冻结前
#     一律由我们改 md 再 sync (防双源漂移); 冻结后再切换 Overleaf 为终稿层
#
# 两篇一个项目 (user 决定 2026-07-27, 方案 a):
#   REALM @ EMNLP 2026 收两篇, 共用一套 ACL 样式 + 一个 paper.bib, 所以放同一个
#   Overleaf 项目。文件名扁平且带 paper 前缀 (main_paperA.tex / paperA_body.tex ...)
#   而不是子目录, 因为 Overleaf 从项目根编译, 扁平布局下 \input 路径不需要改写。
#   **切换编译目标**: Overleaf 菜单 → Main document → main_paperA.tex 或
#   main_paperB.tex。两个 main 共享 acl.sty / acl_natbib.bst / paper.bib / figures/。
#
# 首次配置:
#   1. user 在 Overleaf 新建项目, Menu → Git 拿 URL
#   2. git clone https://git.overleaf.com/<PROJECT_ID> ~/overleaf-aaai27
#      (密码 = Overleaf Account Settings → Git authentication token)
#   3. export OVERLEAF_GIT_DIR=~/overleaf-aaai27 (或写入 scripts/vwa_env_remote.sh)
#   注: clone 目录名仍叫 overleaf-aaai27 是历史遗留 (AAAI-27 撤出 2026-07-22);
#       改名要 user 在 Overleaf 端重建项目, 不值得。
#
# ⚠️ 不同步 main_restructured.tex 那套 (2026-08-04)。该 draft 的唯一真源在 Overleaf
#   项目里, 不在本仓库: 它的 sections/ 是在 Overleaf 端手写的散文, tables/ 带着那边
#   做的人工修复 (7 处 caption / 25 张 table* 转换)。任何"重新生成再拷过去"都会抹掉
#   这两样。本脚本曾短暂加过一个 restructured 分支, 它在首次运行时就覆盖了那边三次
#   提交的散文 (已恢复), 因此整支移除而不是加保护 —— 一条不该存在的路径, 修补它只是
#   让它更难被发现是错的。表的更新按需手工做: 新表追加编号, 改名做文本替换。
#
# 用法:
#   bash scripts/maintenance/overleaf_sync.sh                      # 两篇都同步
#   bash scripts/maintenance/overleaf_sync.sh paperA               # 只同步 A
#   bash scripts/maintenance/overleaf_sync.sh paperA "msg"         # 自定义 commit message
#   SUBMISSION=1 bash scripts/maintenance/overleaf_sync.sh         # 走 --submission 严格门

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
LATEX_DIR="$REPO/docs/checkpoints/paper_drafts/latex"
OL="${OVERLEAF_GIT_DIR:-$HOME/overleaf-aaai27}"

PAPERS=()
case "${1:-}" in
  paperA|paperB|realm) PAPERS=("$1"); shift ;;
  # 2026-08-03: both old drafts were archived to
  # docs/archive/paper_drafts_pre_rewrite_2026-08-03/ and the rewrite happens in realm/.
  # paperA was removed from the Overleaf project; paperB's .tex files are left in place,
  # FROZEN at their 2026-07-28 build, as the advisor's reference copy. Its Markdown source
  # is archived, so `convert.sh paperB` cannot run and must not be attempted — hence the
  # default is realm alone. Passing `paperB` explicitly will fail, deliberately and loudly.
  "") PAPERS=(realm) ;;
  *) echo "✗ 未知参数: $1 (要 paperA / paperB / realm / 空)" >&2; exit 1 ;;
esac
MSG="${1:-sync: REALM drafts $(cd "$REPO" && git rev-parse --short HEAD)}"

if [[ -z "$OL" || ! -d "$OL/.git" ]]; then
  echo "✗ OVERLEAF_GIT_DIR 未设置或不是 git clone (见脚本头部首次配置)" >&2
  exit 1
fi

# Overleaf 端如果有网页编辑未回流, 覆盖会丢它。先拉一次而不是盲推。
echo "[0/3] git pull --ff-only (查 Overleaf 端是否有未回流编辑)"
if ! git -C "$OL" pull --ff-only 2>&1 | tail -2; then
  echo "✗ pull 非 fast-forward: Overleaf 网页端有本地没有的提交。" >&2
  echo "  先人工核对 (git -C $OL log --oneline -5), 不要覆盖。" >&2
  exit 3
fi

SUBMISSION_FLAG=()
[[ "${SUBMISSION:-0}" == "1" ]] && SUBMISSION_FLAG=(--submission)

for PAPER in "${PAPERS[@]}"; do
  BUILD="$LATEX_DIR/build/$PAPER"
  echo "[1/3] convert.sh $PAPER ..."
  if ! (cd "$LATEX_DIR" && bash convert.sh "$PAPER" "${SUBMISSION_FLAG[@]}") \
      > "/tmp/overleaf_sync_convert_$PAPER.log" 2>&1; then
    echo "✗ convert.sh $PAPER 失败, 见 /tmp/overleaf_sync_convert_$PAPER.log" >&2
    tail -5 "/tmp/overleaf_sync_convert_$PAPER.log" >&2
    exit 2
  fi
  grep -E "^(PDF|Reference start page|Visible TODO|Undefined citations):" \
    "/tmp/overleaf_sync_convert_$PAPER.log" || true

  echo "[2/3] 拷贝 $PAPER 产物 → $OL"
  mkdir -p "$OL/figures"
  # per-paper 文件加前缀; 共享文件 (样式/bib/图) 不加 —— 两篇的副本逐字节同源
  # (convert.sh 从同一个 LATEX_DIR + 同一个 paper.bib 生成), 所以互相覆盖无害。
  sed -e "s|{title_generated\.tex}|{${PAPER}_title.tex}|g" \
      -e "s|{abstract_generated\.tex}|{${PAPER}_abstract.tex}|g" \
      -e "s|{body_generated\.tex}|{${PAPER}_body.tex}|g" \
      -e "s|{limitations_generated\.tex}|{${PAPER}_limitations.tex}|g" \
      -e "s|{appendix_generated\.tex}|{${PAPER}_appendix.tex}|g" \
      "$BUILD/main.tex" > "$OL/main_$PAPER.tex"
  cp "$BUILD/title_generated.tex"    "$OL/${PAPER}_title.tex"
  cp "$BUILD/abstract_generated.tex" "$OL/${PAPER}_abstract.tex"
  cp "$BUILD/body_generated.tex"     "$OL/${PAPER}_body.tex"
  for optional_part in limitations appendix; do
    if [[ -f "$BUILD/${optional_part}_generated.tex" ]]; then
      cp "$BUILD/${optional_part}_generated.tex" "$OL/${PAPER}_${optional_part}.tex"
    else
      rm -f "$OL/${PAPER}_${optional_part}.tex"
    fi
  done

  # Every \input the rewritten main still points at must exist in the project,
  # or Overleaf fails to compile on a file the sed rename missed. This caught a
  # missing limitations rename on 2026-07-27, after it had already been pushed.
  while read -r referenced; do
    if [[ ! -f "$OL/$referenced" ]]; then
      echo "✗ main_$PAPER.tex references $referenced, which is not in the project" >&2
      exit 4
    fi
  done < <(grep -oE '\\input\{[^}]+\}' "$OL/main_$PAPER.tex" | sed 's/.*{//; s/}//')
  cp "$BUILD/acl.sty" "$BUILD/acl_natbib.bst" "$BUILD/paper.bib" "$OL/"
  cp "$BUILD"/figures/*.pdf "$OL/figures/" 2>/dev/null || true
done

# 旧 AAAI 产物集 (aaai2027.sty/.bst + 无前缀 main.tex) 与两个新 main 同时存在时,
# Overleaf 侧会看到多个 \documentclass 并挑错主文件。显式清掉。
rm -f "$OL/aaai2027.sty" "$OL/aaai2027.bst" "$OL/main.tex" \
      "$OL/title_generated.tex" "$OL/abstract_generated.tex" "$OL/body_generated.tex"

echo "[3/3] commit + push"
cd "$OL"
git add -A
if git diff --cached --quiet; then
  echo "· 无变化, 跳过 push"
else
  git commit -q -m "$MSG"
  git push -q origin master 2>/dev/null || git push -q origin main
  echo "✓ pushed: $MSG"
  echo "  Overleaf 菜单 → Main document → main_realm.tex (当前) 或 main_paperB.tex (冻结参照)"
fi
