# LaTeX 转换工作流（2026-07-14 dry-run）

唯一入口：

```bash
./convert.sh
```

脚本每次重建 `build/`，自动完成 HTML comment 剥离、占位符 `\todo`
标记、pandoc `--natbib` 转换、四张 Markdown 表的 `booktabs table*`
转换、F1/F2 复制和 `latexmk` 编译。产物是 `build/main.pdf`；全局
`.gitignore` 的 `build/` 规则已覆盖该目录。

`paper.bib` 的 `note` 字段含内部文献消化文字，标准 BibTeX style 会将其
原样打印；脚本仅在生成的 `build/paper.bib` 中剥掉这些单行字段，不改
canonical Bib 文件。

若本目录同时存在 `aaai27.sty` 与 `aaai27.bst`，脚本使用官方
`submission` 模式；否则使用标准 `article[twocolumn]` + `plainnat`
代理。代理 PDF 只证明转换链可编译，页数不等于正式 AAAI 页数。

提交前仍须确认：官方 author kit 已放入本目录、全部 `\todo` 清零、
F2 不含 `INTERIM/PARTIAL_DATA` 水印、BibTeX 无 warning、匿名信息清零。
