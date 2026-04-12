# B1 Reddit Findings

> 占位 — Reddit B1 三模式全部完成后填充跨模式对比分析。

## 当前状态

DOM 运行中，SoM / Vision 待启动。

## 早期发现

### Comment 自链接死循环（DOM task 0/1/3）

详见 [B1_DOM_digest.md](B1_DOM_digest.md)。

核心问题：Reddit Postmill 的帖子页面即评论页面，"N comments" 链接指向当前页面自身。Agent 不理解「已到达目标」，反复点击同一链接。§33 新增的 URL stuck 检测（5 次连续 click 同 URL 早停）应能覆盖此模式。

### 与 Classifieds 的预期差异

| 维度 | Classifieds | Reddit（预期） |
|------|------------|---------------|
| 站点结构 | 分类→列表→详情 | 论坛→帖子→评论 |
| 主要导航模式 | 搜索/筛选 + 翻页 | 浏览 + 评论跳转 |
| Visual task 比例 | 67% (162/234) | 待统计 |
| 自链接问题 | 少见 | 已观察到（comment 链接） |
