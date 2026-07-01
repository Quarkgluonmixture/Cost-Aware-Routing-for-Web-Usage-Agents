# LaTeX 转换工作流 (队列⑩, 2026-07-02)

## 前置（一次性）
1. AAAI-27 author kit: 官网 authorkit 下载 `aaai27.sty` + `aaai27.bst` 放本目录
   （⚠️ kit 未入库前 skeleton 编译不过——结构已锁, 不 block 写作）。
2. `mkdir -p figures && cp ../../../../..//results/phantom_paper/figures/fig_f1_diamond_schematic.pdf figures/`
   （F2/F3 同理; **verdict-day 重生成后再拷**, INTERIM 水印版禁入）。
3. bib: `cp ../../paper.bib .`（或 symlink; kit 用 natbib + aaai27.bst）。

## md → tex 转换（verdict slots 填完之后才做）
```bash
# 1. 剥 HTML comments (checklist 块等) — 注意别在 md 里内联这个命令 (item-7 教训)
# 2. pandoc 转 body (natbib citation)
pandoc ../aaai27_main.md -f markdown -t latex --natbib --wrap=none \
  -o body_generated.tex
# 3. 手工整入 skeleton 各 \section (pandoc 输出是平铺的, header 层级会带
#    \section/\subsection, 但 Table 1-4 的 markdown 管道表转出来是 longtable —
#    需手工换成 booktabs tabular; citation key [@x] → \citep{x} pandoc 自动)
```

## 提交前 checklist (与 aaai27_main.md 末尾 checklist 联动)
- [ ] `[submission]` 模式编译 (匿名) + 页数 ≤7 正文 (refs 不计)
- [ ] 全部 `TODO:` 清零; INTERIM 水印图 0 张 (grep "INTERIM" 编译日志/图源)
- [ ] `bibtex` 0 warning (paper.bib key 与 --natbib 输出对齐; nikankin2025sametask 留 rebuttal 不入正文)
- [ ] `\graphicspath` 下图全为 verdict-day 重生成版
- [ ] grep 匿名违规: host 名 (spark/condense/quark/myriad) / 用户名 / OSF 非匿名链接
- [ ] reproducibility checklist 按 kit 要求填 (aaai27_main.md # Reproducibility statement 为源)
