 §106 VWA / WebArena Coordinate-Based Click Dispatch Bug — Complete Overview
                                                                                                                   
  TL;DR                                                  
                                                                                                                   
  WebArena 和 VisualWebArena 上游代码在处理 click [element_id] 这条 canonical action 时，把 AXTree 节点的          
  union_bound（外接矩形）几何中心当作点击目标，并通过 page.mouse.click(x, y) 直接发送坐标点击。问题是当 AXTree
  节点对应的是父容器（<li class="listing-card"> / <dl class="filter-options"> / <h1                                
  class="submission__title">），或对应的是 inline 多行 <a> 元素时，几何中心经常不落在真正可点击的 <a> hit-area 
  内，被 document.elementFromPoint 解析为父元素，silent failure（无 navigation，无错误抛出，agent
  看到的下一帧观测一字不差）。

  我们在 4 个 paper-grade run（B0/B1 × classifieds/reddit ≈ 1730 episodes）上扫描发现 93 个 episode 命中此         
  signature，其中 90 个 (97%) 失败。Bug mode-asymmetric——DOM 命中率系统性高于 SoM（B0 cls 上 6.5×），影响所有用
  element_id-based click 的模式（DOM、SoM、Phantom-text、Phantom-SoM、Phantom-prompt），唯独 Vision 不受影响（agent
   自己输出归一化坐标）。                                

---
  1. Bug 定义（精确）

  触发条件

  - Action format: {"action_type": "click", "element_id": N}                                                       
  - AXTree 节点 N 的 union_bound 是父容器的外接矩形而非真 <a> 的位置；或节点是 inline 多行 <a> 文本，bbox 跨越 line
      gap                                                                                                             
                                                         

  表现                                                                                                             
                                                         
  - action_success=False
  - page_changed=False
  - page_change_reasons=[]                                                                                         
  - text_similarity≈1.0（DOM 一字不差）
  - env_step_ms 5000-7000ms（Playwright navigation 等待 timeout）                                                  
  - 连续 ≥2 次相同 click 后触发 cycle/no_progress 早停                                                             
                                                                                                                   

  ▎ Explanatory sidebar： AXTree（Accessibility Tree）是浏览器为屏幕阅读器构建的语义树，节点用文本 + 角色（link /  
  ▎ button / heading）标注。WA/VWA 把 AXTree 当 agent 的"地图"，每个节点分配一个数字 ID 让 agent 引用。问题是      
  ▎ AXTree 的语义分组未必和 visual layout 对齐——一个 link 节点可能从 DOM 上对应一个 <a>，但 getBoundingClientRect  
  ▎ 返回的 bbox 受 CSS 影响，常常涵盖父级 <li> 的整个 padding / image / 文字 union 区域，而非 <a> 元素的真实文字 
  ▎ hit-area。

---
  2. Root Cause（代码路径）
                                                                                                                   

  上游 web-arena-x/visualwebarena 与 web-arena-x/webarena 同款，未修
                                                                                                                   
  Layer 1 — bbox 来源错误（browser_env/processors.py:786-795）：                                                   
                                                                                                                   
  def get_element_center(self, element_id: str) -> tuple[float, float]:                                            
      node_info = self.obs_nodes_info[element_id]                                                                  
      node_bound = node_info["union_bound"]   # ← 父容器的总 bbox                                                  
      x, y, width, height = node_bound                                                                             
      center_x = x + width / 2                                                                                     
      center_y = y + height / 2                          
      return (center_x / viewport_w, center_y / viewport_h)                                                        
                                                                                                                   
  Layer 2 — 用坐标点击而非 locator 点击（browser_env/actions.py:1305-1308）：                                      
                                                                                                                   
  case ActionTypes.CLICK:                                                                                          
      if action["element_id"]:                                                                                     
          element_center = obseration_processor.get_element_center(element_id)
          execute_mouse_click(element_center[0], element_center[1], page)                                          
          # → page.mouse.click(x*viewport_w, y*viewport_h)                                                         
          # 不 scroll-into-view、不 actionable check、不 retry、不 element resolution                              
                                                                                                                   
  对比同文件正确路径（role+name 路径走 locator.click()，pw_code 路径走 execute_playwright_click(locator_code)，都用
   Playwright 标准 locator API，自动处理 scroll-into-view / actionability / element resolution）。只有 element_id  
  路径是 broken 的。                                                                                               
                                                         
  ▎ Explanatory sidebar — Playwright locator.click() vs page.mouse.click()： locator.click() 是高级 API：先用      
  ▎ selector 找到元素 → 等可见且稳定 → 自动 scroll into view → 计算保证落在元素内的合法点 → 重试可达 timeout 
  ▎ 次数。page.mouse.click(x, y) 是底层                                                                            
  ▎ API：在那个像素位置按下鼠标，谁碰到谁就接收点击事件，没有任何元素感知。WA/VWA 的 element_id click 路径选了底层 
  ▎ API，是性能/精确度的权衡——但精确度只在"bbox 真正对应叶子元素 hit-area"成立时才高，对现代 CSS
  ▎ 卡片布局基本不成立。

---
  3. Discovery Context
                      

  起因：用户在 B0_dom_shopping_20260428（466 task paper-grade run）上 debug task 0 失败，发现 swatch radio click
  漏报（→ §105 swatch bug，dict-key collision，已修）。然后在 task 5 step 3 发现性质完全不同的另一种 click silent  
  failure：
                                                                                                                   
  - Task 5 intent: "show me the most expensive yellow product from Dried Fruits & Vegetables category"             
  - Step 3-5 连续 3 次 click element 10752（侧栏 "Grocery & Gourmet Food" 链接），全部 silent fail
  - Playwright 实证：用 .auth/shopping_state.json 重现到 step 3 状态后：                                           
    - <a>.getBoundingClientRect() 返回 [20, 95.09, 169, 39] ✅ AXTree 没说谎                                       
    - document.elementFromPoint(104.5, 114.59) → <li class="item"> ❌ 父元素                                       
    - page.locator('a:has-text("Grocery")').click() → 成功导航 ✅                                                  
                                                                                                                   

  关键判断：bbox 是对的，几何中心计算也是对的，问题在于"几何中心是否落在 <a> 的 inline content"——对 inline display 
  的多行链接，line gap 处的中心点会命中父 <li>。                                                                   
                                                                                                                   
---
  4. Empirical Blast Radius（4 paper-grade runs）
                                                                                                                   

  扫描 signature: 连续 ≥2 click steps, action_success=False, page_changed=False, page_change_reasons=[],
  element_bbox h≥25 & w≥80（排除 swatch 小 icon），env_step≥4000ms（排除快速 click）：                             
                                                         
  ┌────────┬───────────────────┬────────┬─────┬─────┐                                                              
  │  Run   │   Episodes 命中   │ 失败率 │ DOM │ SoM │    
  ├────────┼───────────────────┼────────┼─────┼─────┤                                                              
  │ B0 cls │           7 / 234 │   100% │   6 │   1 │    
  ├────────┼───────────────────┼────────┼─────┼─────┤
  │ B0 red │          17 / 210 │   100% │  10 │   7 │                                                              
  ├────────┼───────────────────┼────────┼─────┼─────┤
  │ B1 cls │           8 / 234 │   100% │   5 │   3 │                                                              
  ├────────┼───────────────────┼────────┼─────┼─────┤                                                              
  │ B1 red │          61 / 210 │    93% │  29 │  32 │
  ├────────┼───────────────────┼────────┼─────┼─────┤                                                              
  │ 合计   │ 93 / 1730 (~5.4%) │    97% │  50 │  43 │    
  └────────┴───────────────────┴────────┴─────┴─────┘                                                              
                                                         
  shopping 暂未扫（runner 还在跑 paper-grade clean re-run），估计另有 ~5-15 ep。                                   
                                                         
  至少 4 种亚模式（全部已 Playwright 实证）                                                                        
                                                         
  ┌──────────────────┬───────────────────────────────────────┬─────────────────────────────┬───────────────────┐   
  │      亚模式      │                 触发                  │    elementFromPoint 命中    │ 实例（task 编号） │
  ├──────────────────┼───────────────────────────────────────┼─────────────────────────────┼───────────────────┤   
  │ 1. inline        │ 链接文字换行，bbox 跨多行             │ 父 <li class="item">        │ shopping task 5   │
  │ multi-line <a>   │                                       │                             │                   │
  ├──────────────────┼───────────────────────────────────────┼─────────────────────────────┼───────────────────┤   
  │ 2. Magento       │ <li.listing-card>                     │ 父 <li.listing-card>        │ classifieds task  │   
  │ listing card     │ 含图+价+标题，AXTree 报父 bbox        │                             │ 65/123/204        │   
  ├──────────────────┼───────────────────────────────────────┼─────────────────────────────┼───────────────────┤   
  │ 3. Reddit post   │ <h1.submission__title>                │ 命中 <a> 但                 │                   │
  │ title block      │ 含标题+username 行                    │ href→image，可能            │ reddit task 169   │
  │                  │                                       │ popup-blocked               │                   │   
  ├──────────────────┼───────────────────────────────────────┼─────────────────────────────┼───────────────────┤
  │ 4. Reddit        │ <span.subscribe-button__label> inside │ 命中正确 button，但         │ reddit task 130   │   
  │ subscribe button │  <button>                             │ form/CSRF 问题              │                   │
  └──────────────────┴───────────────────────────────────────┴─────────────────────────────┴───────────────────┘   
                                                         
  ▎ Explanatory sidebar — 为什么这些 signatures 一致失败？ 这些 web 框架（Magento Luma 主题、Postmill）都是 modern 
  ▎ CSS card-based UI——卡片用 <li> / <div> 做容器，里面嵌套图、文、链接，让卡片整体感觉"可点"但只有标题 <a> 
  ▎ 是真链接。AXTree 把整个卡片标记成 link 节点（因为 ARIA role），bbox                                            
  ▎ 是卡片整体大小，几何中心落在图片或文字间空白。用 BrowserGym 风格的 bid-locator 点击就能绕开，因为 locator 直接 
  ▎ query 真 <a> 而不是几何点。

---
  5. Mode 不对称性（paper-relevant）
                                                                                                                   

  ┌────────┬────────────┬────────────┬──────┐
  │  Run   │ DOM 命中率 │ SoM 命中率 │ 比值 │                                                                      
  ├────────┼────────────┼────────────┼──────┤            
  │ B0 cls │       2.6% │       0.4% │ 6.5× │                                                                      
  ├────────┼────────────┼────────────┼──────┤            
  │ B0 red │       4.8% │       3.3% │ 1.5× │                                                                      
  ├────────┼────────────┼────────────┼──────┤                                                                      
  │ B1 cls │       2.1% │       1.3% │ 1.6× │                                                                      
  ├────────┼────────────┼────────────┼──────┤                                                                      
  │ B1 red │      13.8% │      15.2% │ 0.9× │            
  └────────┴────────────┴────────────┴──────┘                                                                      

  这是 paper-grade 风险的核心：bug 不是均匀拉低所有 mode 的 SR，而是差异化污染模式间的相对比较。具体污染：         
  - §23 "Mirage Effect" 解释（DOM-SoM gap）              
  - §103 Phantom-SoM "drop-one oracle 1.7-3.3pp" 主 hook                                                           
  - §103 "30-40% non-overlap task pool" 论证             
  - Section 6 router signal AUROC 训练数据                                                                         
                                                         
---
  6. Literature Status（Deep Research 结论）             
                                                                                                                   

  已有
                                                                                                                   
  - WebSuite (Li & Waldo 2024, arxiv 2406.01623)：观察到 SeeAct 在 E2E navigational task 0% 成功率（vs 隔离 click  
    72%），明确说"agent tries to click the larger container <div> rather than the actual <a> link"。症状级 cite-able 
    证据。                                                                                                           
  - BrowserGym (Drouin et al, OpenReview)：用 bid 属性注入 + locator.click("[bid='N']") 整体绕开，但 design
    rationale 没明确点名 union_bound math 是问题                                                                     
  - VisualWebArena paper 自己（Koh et al. ACL 2024）：在 evaluation metric 设计里承认 "center of bbox 是不可靠的" —
      但 execution pipeline 仍用同一 broken math（paradox）                                                           
                                                         

  缺失（我们 niche）                                                                                               
                                                         
  - 没人把 WebSuite 症状 trace 到 processors.py:get_element_center 这条具体 math                                   
  - 没人给 WA/VWA fork 实现一个 drop-in fix 让原 prompt format 不变情况下绕开
  - 没人量化跨 site / 跨 mode 的 differential bias（我们的 4-run scan 是首个）                                     
  - 上游 issue tracker、PR、commit log 全部 NO_EVIDENCE                                                            
                                                                                                                   

  ▎ Explanatory sidebar — 为什么这是个有 publication value 的 niche？ Reviewer 喜欢三类 contribution：(1) 新       
  ▎ algorithm，(2) 新 benchmark，(3) methodology rigor——揭示 community 共识里的隐性 bias，提供可复现 fix。这条 bug 
  ▎ 属于第三类。WebSuite 已经 plant the flag（"we see the symptom"），BrowserGym 提供了正确路径但没明说为什么，VWA 
  ▎ 自己 contradiction-aware but not contradiction-fixed。我们如果同时做"causal trace + magnitude quantification + 
  ▎ drop-in fix"，是把零散信号综合成一个完整 contribution——这是非典型但 reviewer-friendly 的 paper 价值。

---
  7. Fix Recipe
               

  首选方案：BrowserGym 风格 bid 注入 + locator click 替换 mouse click：
                                                                                                                   
  1. JS 注入 data-vwa-bid="${id}" 到每个 AXTree 节点的对应 DOM 元素（在 processors.py 抓 AXTree 时同步给 element 加
      attribute，~30 行 JS）                                                                                          
  2. actions.py:1305-1308 改写：                                                                                   
                                                                                                                   

  case ActionTypes.CLICK:
      if action["element_id"]:                                                                                     
          element_id = action["element_id"]              
          try:
              page.click(f'[data-vwa-bid="{element_id}"]', timeout=5000)
          except PlaywrightTimeoutError:                                                                           
              # fallback to coord click for legacy compatibility                                                   
              element_center = obseration_processor.get_element_center(element_id)                                 
              execute_mouse_click(element_center[0], element_center[1], page)                                      
                                                                                                                   
  3. 下沉到 P79 fork (Quarkgluonmixture/visualwebarena p79-patches)，加 commit                                     
  4. Optional：PR 上游（连同 audit doc + reproduce 步骤）                                                          
                                                                                                                   

  备选：保守 fallback（不动 mouse click）                                                                          
                                                                                                                   
  # 用 elementFromPoint 校验 click 是否落在 element_id 对应节点                                                    
  expected_node = obseration_processor.get_node_by_id(element_id)                                                  
  hit_node = page.evaluate(f"document.elementFromPoint({cx}, {cy})")                                               
  if hit_node != expected_node and not is_descendant(expected_node, hit_node):                                     
      # fallback locator click                                                                                     
      page.locator(f"[data-vwa-bid='{element_id}']").click()                                                       
  else:                                                                                                            
      execute_mouse_click(cx, cy, page)                                                                            
                                                                                                                   
  更保守但代码复杂；首选方案更干净。                                                                               
                                                                                                                   
---
  8. Re-run Scope（Decision Pending）                    
                                     

  必须 rerun 的 cells（受影响 mode）
                                                                                                                   
  ┌─────────────┬───────────────────────────────────┬─────────────────────────┐
  │    Site     │             B0 modes              │        B1 modes         │                                    
  ├─────────────┼───────────────────────────────────┼─────────────────────────┤
  │ classifieds │ DOM, SoM, P-text, P-SoM           │ DOM, SoM, P-text, P-SoM │
  ├─────────────┼───────────────────────────────────┼─────────────────────────┤
  │ reddit      │ DOM, SoM, P-text, P-SoM, P-prompt │ DOM, SoM, P-text, P-SoM │                                    
  ├─────────────┼───────────────────────────────────┼─────────────────────────┤                                    
  │ shopping    │ DOM, SoM, P-text, P-SoM           │ DOM, SoM, P-text, P-SoM │                                    
  └─────────────┴───────────────────────────────────┴─────────────────────────┘                                    
                                                         
  14 cells 全 rerun，Vision 不动（不受影响）。                                                                     

  估算                                                                                                             
                                                         
  - ~10800 episodes total                                                                                          
  - B1 GPU: ~90 hours（Myriad 独占）
  - B0 API: ~$50-80                                                                                                
  - Wall time: ~3-4 days on Myriad（不抢资源）                                                                     
                                                                                                                   

  同时合并 §105 swatch fix                                                                                         
                                                                                                                   
  - §105: shopping 11 ep 已确认受影响（task 0/46/245/262/281/288/329/349/383/415/458）                             
  - §106: 跨 3 site                                      
  - 两个 fix 合并成一个 commit + 一个 rerun wave，增量成本相对 §105 单独 rerun 几乎为 0                            
                                                                                                                   
---
  9. Paper Framing Options                                                                                         
                                                         

  ┌──────────────────────────┬─────────────────────────────┬─────────┬────────────────────────────────────────┐
  │          Option          │           在哪写            │  字数   │                  风险                  │    
  ├──────────────────────────┼─────────────────────────────┼─────────┼────────────────────────────────────────┤
  │ A. Methodology           │ Section 4 (Limitations) 或  │         │                                        │    
  │ contribution（推荐）     │ Section 8 (Discussion) 的   │ 500-700 │ 需要修 + rerun，ETA 1-2 周             │
  │                          │ sub-section                 │         │                                        │    
  ├──────────────────────────┼─────────────────────────────┼─────────┼────────────────────────────────────────┤
  │ B. Limitation note 不修  │ Section 4 footnote 或       │ 100-200 │ mode-asymmetric 偏差污染 phantom       │    
  │                          │ Section 8 一段              │         │ hook，reviewer 可能要求 rerun          │
  ├──────────────────────────┼─────────────────────────────┼─────────┼────────────────────────────────────────┤    
  │ C. 修了不写              │ —                           │       0 │ 浪费 contribution                      │
  │                          │                             │         │ opportunity，方法学诚实度受损          │    
  └──────────────────────────┴─────────────────────────────┴─────────┴────────────────────────────────────────┘
                                                                                                                   
  推荐 A：fix + rerun + 写 500-700 字 + 1 个 before/after SR table + appendix 给 1-page diff。Cite WebSuite        
  作为前人观察。
                                                                                                                   
  ▎ Explanatory sidebar — 为什么 B 的风险高于看起来： "Limitation note 不修"逻辑是"prior literature 也有这个       
  ▎ bug，所以我们 apples-to-apples"，但 §106 命中率在 cls 的 DOM-SoM 间是 
  ▎ 6.5×，不对称是结构性的，不是简单平移。Reviewer 会问："你们的 phantom 1.7-3.3pp gap 在 ~5% bug noise            
  ▎ 下统计显著吗？" 没有 fix 后的对比数字，只能答"不知道"。这是 desk reject 风险。

---
  10. 当前状态（writing-time snapshot）
                                                                                                                   
  - ✅ §106 root cause 已 Playwright 实证
  - ✅ 4-run blast radius scan 完成（logs/codex/inline_click_bug_scan.json）                                       
  - ✅ Deep Research 结论已归档（docs/literature/Deep Analysis of WebArena and VisualWebArena Coordinate-Based     
    Click Discrepancies and GUI Grounding Artifacts.md）                                                             
  - ⏳ §105 swatch fix 已应用（p79/experiment/state_change.py:_key），unit test 通过                               
  - ❌ §106 fix 未应用（待 dom shopping 整轮 debug 结束后合并应用）                                                
  - ❌ 14-cell paper-grade rerun 未启动（待 fix）                                                                  
  - ❌ Audit doc docs/analysis/cross_sites/click_dispatch_bug_audit.md 未写                                        
                                                                                                                   
---
  11. 关键文件路径（让另一个 conversation 直接定位）                                                               
                                                                                                                   
  # Bug 代码（上游 + fork 同款）
  external/visualwebarena/browser_env/processors.py:786-795   # get_element_center                                 
  external/visualwebarena/browser_env/actions.py:1305-1308    # ActionTypes.CLICK with element_id                  
  external/visualwebarena/browser_env/actions.py:954-962      # execute_mouse_click                                
                                                                                                                   
  # 已修的相邻 bug（同一 click 路径下游）                                                                          
  p79/experiment/state_change.py:_key                          # §105 swatch fix                                   
  tests/test_state_change.py                                   # 5 unit tests                                      
                                      
  # Empirical 数据                                                                                                 
  results/visualwebarena/phase1/B0_3mode_classifieds_20260413/  # B0 cls (7 ep affected)
  results/visualwebarena/phase1/B0_3mode_reddit_20260422/        # B0 red (17 ep)                                  
  results/visualwebarena/phase1/B1_3mode_classifieds_20260413/  # B1 cls (8 ep)                                    
  results/visualwebarena/phase1/B1_3mode_reddit_20260413/       # B1 red (61 ep)                                   
  logs/codex/inline_click_bug_scan.json                         # 全 93 ep 详情                                    
                                                                                                                   
  # Documentation                                                                                                  
  docs/literature/Deep Analysis of WebArena and VisualWebArena Coordinate-Based Click Discrepancies and GUI        
  Grounding Artifacts.md                                                                                           
  docs/analysis/cross_sites/swatch_form_change_audit.md          # §105 swatch（不同性质 bug）
  docs/checkpoints/实验笔记.md  §105                              # swatch fix chronicle     