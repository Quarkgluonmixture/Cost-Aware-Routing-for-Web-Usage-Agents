# Runbook —— 演讲那 10 分钟，手要做的（16 Sep · 14:45 slot）

_只放操作：环境、标签页、别碰清单、出事怎么办。要念的在 `rehearsal-script.md`，展板 live 页的运维在 `../demo/README.md` → *Live*。_

## 开讲前 5 分钟（14:40，带 MacBook 从展板走到讲台）

_演讲只用 **MacBook + Chrome**。quark 全天留在展板（live 页的 ssh 隧道只在 quark 上能开），不要搬它。_

| # | 做 | 确认 |
|---|---|---|
| 1 | MacBook 接投影（机身没有 HDMI 口就用 USB-C→HDMI 转接头）；⌘F1，或 控制中心 → 屏幕镜像 → 选「镜像」；看一眼是 16:9 还是 16:10 | 片子第 1 张四边不裁 |
| 2 | **Chrome**（不用 Safari，只测过 Chromium）**只留三个标签页**，从左到右：① `talk/index.html#opening` ② `demo_portable.html?task=130&autoplay=0` ③ `talk/fallback.html` | 其余标签页全关；书签栏收起 |
| 3 | 浏览器缩放保持 **100%**（⌘0）；全屏 ⌃⌘F，并在菜单「显示」里取消「全屏模式下始终显示工具栏」。⛔ **不要放大**：demo 里只有截图会伸缩，1080p 上放大到 125% 截图只剩一半高，150% 时截图整个消失（09-13 实测）。片子第 2 张的 demo 已经按区域整体缩放，放大浏览器也不会让它变大 | 第 2 张三栏里的网页截图清楚可见 |
| 4 | ① 带 `#opening` 刷新一次（⌘R） | 停在 `opening`，不跳 |
| 5 | ② 确认停在 **task 130 · step 1/…**，Play 按钮显示 *Play* 不是 *Pause* | 三栏第 0 帧都在 |
| 6 | 控制中心 → 专注模式 → 勿扰；电源接上；静音；终端里跑 `caffeinate -d`（不熄屏，讲完 ⌃C） | 无弹窗 |
| 7 | 手机计时器 9:30 | 放讲台上 |

展板旁的 quark 继续自动播放，**不要动它**。Mac 按键对照：切标签页 ⌃Tab · 参考页（备用录屏）按 R · 收尾页 Fn+→（= End）· 退出全屏 ⌃⌘F。

## 台上的动线（页面按 id，对着台本的段落）

| 段落 | 手 |
|---|---|
| opening | ① 停在 `opening`。说完最后一句 → 按 → 到 `demo` |
| demo | 点一下 iframe 里的空白处让它拿到焦点；只按 **→**，共 10 下（READ 9 步 + 1 下看结束态），再按一次 → 自动翻到 `question`。⛔ 不按 space，⛔ 不按 1/2/3/4 |
| question → not-yet | 每段按一次 →：`question` → `behaviour` → `failure` → `hindsight` → `learned` → `why` → `not-yet`；`learned`「only one of eight」后**不翻页，停一下** |
| close | 按 → 到 `close`，收 |

参考页 `reference`（39 秒录屏）不在主讲流程里：按 **R** 打开，再按 → 回到 `close`。End（Mac 上 Fn+→）直达 `close`。

## 别碰清单

- 不切编辑器、不 sign in、不搜索、不点导航
- 演讲里**不进 live 页**（`4`），live 只在展板上
- 不在台上开新标签页；不动展板那台电脑
- 演讲前一天之后不重建 demo、不改片子（改一次就要重跑 `check_talk.py` 和三遍彩排里的点击那遍）

## 出事了怎么办

| 症状 | 做 | 说 |
|---|---|---|
| iframe 白屏 / 不步进 | ⌃Tab 切标签页 ②（同一个演讲版 demo，独立页面） | 什么都不说，继续第二幕 |
| ② 也不行 | 切标签页 ③ 兜底页，← → 翻截图 | *"I've got these captured."* 然后按截图讲 |
| 投影只认主办方电脑 / MacBook 接不上 | U 盘里 `showcase/` 文件夹（`demo_portable.html` 与 `talk/` 同级）；任何电脑用 Chrome 打开 `talk/index.html` | — |
| 浏览器崩 | 重开浏览器只开 ③ | 一句带过，再也不提 |

⛔ **永远不说 "it worked this morning"。** ⛔ 台上不调试。

## 会后

- 一小时内把 repo 链接（海报页脚那个 QR）和 `demo_portable.html` 发给问过的人
- 撤展时 DGX 侧：`fuser -k 8799/tcp`；`docker compose -f deliverables/showcase/demo/live/site-compose.yml down`
- LOG：结果 + 被问的问题 + 哪一拍没立住 → 笔记 §（当天）
