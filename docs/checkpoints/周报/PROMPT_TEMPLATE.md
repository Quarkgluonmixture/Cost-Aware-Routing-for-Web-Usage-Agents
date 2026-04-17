
```
你是一个前端可视化专家。我需要你把下面的 Markdown 周报转换为一个 React 单页仪表盘（Single Page Dashboard），用于周会投屏展示。

技术栈：React + Tailwind CSS + Recharts + Lucide React Icons。输出一个完整的 App.jsx 文件。

### 核心原则：严格对齐周报内容

**这是最重要的要求。** 可视化必须与 Markdown 周报内容 100% 对齐：

- **不添加**：不要编造或推测任何周报中没有的数据、结论、描述。如果周报没提到某个数字或观点，Dashboard 中也不能出现。
- **不遗漏**：周报中的每个 section、每个表格、每个要点都必须在 Dashboard 中有对应呈现。不要跳过任何内容。
- **不改写**：关键数字（SR、p值、AUROC、成本等）必须原样保留，不要四舍五入或换算。结论措辞忠于原文，可以英译但不要改变语义。
- **不重排**：保持周报的 section 顺序（## 1 → ## 2 → ...），子节顺序也一致。
- **引用检查**：生成完成后，逐段对照周报原文，确认没有遗漏或偏差。

### 设计要求

1. **布局**：单页纵向滚动，max-width 1400px 居中，白色卡片 + 浅灰背景。适合投影仪/大屏展示。
2. **字号**：投屏友好。正文至少 text-base(16px)，标题 text-2xl+，表格 text-sm+，脚注 text-sm。不要用 text-xs。
3. **配色**：indigo 为主色调，slate 为中性色。成功=emerald，警告=amber，错误=red，信息=blue。
4. **组件**：
   - 每个大节（##）用 `SectionHeader` 组件（编号圆圈 + 标题 + icon）
   - 关键指标用 `MetricBox` 组件（标题 + 大号数字 + 注释）
   - 数据对比用 Recharts `BarChart`（支持分组柱状图）
   - 表格用 Tailwind 样式的 HTML table（rounded border + striped rows）
   - 状态列表用卡片 + 颜色编码 border
   - Disclaimer/Warning 用 amber 色 alert box
5. **内容映射规则**：
   - Markdown 的 `## N. 标题` → SectionHeader num={N}
   - Markdown 的 `### N.M 子标题` → 卡片内 h3
   - Markdown 表格 → 如果有数值对比（SR、AUROC 等），优先用 Recharts 图表 + 右侧统计卡片；否则用 HTML table
   - 要点列表 → 视内容选择：数字指标用 MetricBox grid，定性描述用卡片 + icon
   - `> blockquote` 引用 → amber/blue alert box
   - 加粗关键结论保留 `<strong>`
6. **数据外置**：所有周报数据（SR 数字、表格行、状态列表等）定义在文件顶部的 const 变量中，与 JSX 模板分离。这样下周只需改数据，不用改模板结构。
7. **响应式**：`grid-cols-1 md:grid-cols-2 lg:grid-cols-12` 等，手机和大屏都可用。
8. **不需要**：路由、状态管理、API 调用、动画。纯静态展示。

### 风格参考

- Header 区域：项目标签（P79 Weekly Report）+ 大标题 + 日期/阶段/状态/模型 四个 metadata tag + 数据声明 disclaimer
- Section 之间 `mt-10` 间距
- 最后的"下周重点"用 indigo-900 深色卡片 + 白色文字 + 编号圆圈
- Scaffold 修复表格和执行状态并排显示（lg:grid-cols-2）
```

---
4. Gemini 生成 App.jsx 后，替换 `weekly-dashboard/src/App.jsx`
5. 在 `weekly-dashboard/` 目录下运行 `npx vite build`
6. 右键 `周报/dashboard.html` → Open with Live Server 查看

