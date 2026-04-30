import React, { useState } from 'react';
import {
  Sparkles,
  Layers,
  TrendingUp,
  Zap,
  GitBranch,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  DollarSign,
  Cloud,
  Cpu,
  ChevronDown,
  ChevronRight,
  ExternalLink,
  Quote,
} from 'lucide-react';

// Phantom-SoM Advisor Meeting Brief (2026-04-30)
// Companion to: docs/reference/PHANTOM_SOM_ADVISOR_MEETING_BRIEF.md

const HERO_PROPS = [
  { label: 'cost ≈ DOM', detail: '[SOM_MARKS] = AXTree regex filter, no extra LLM call', icon: DollarSign, color: 'green' },
  { label: 'latency 50% lower', detail: 'cls SoM p95 74s vs Phantom-SoM 18.2s (4× faster)', icon: Zap, color: 'amber' },
  { label: 'AUROC ≥ baseline', detail: 'red P-text verbalized 0.793 (5-mode highest)', icon: TrendingUp, color: 'sky' },
  { label: 'drop-one 1.7-3.3pp', detail: 'red P-SoM 3.33pp; cls 2.56pp (Phantom unique)', icon: GitBranch, color: 'violet' },
];

const SIX_MODES = [
  { name: 'DOM', text: 'AXTree (hierarchical)', prompt: 'DOM-prompt', image: '❌', role: 'baseline (cheapest text)' },
  { name: 'P-text', text: '[SOM_MARKS] (flat)', prompt: 'DOM-prompt', image: '❌', role: 'axis 1 swap', highlight: true },
  { name: 'P-SoM', text: '[SOM_MARKS] (flat)', prompt: 'SoM-prompt', image: '❌', role: 'axis 1+2 swap (★ Phantom-SoM)', highlight: true },
  { name: 'SoM', text: '[SOM_MARKS] (flat)', prompt: 'SoM-prompt', image: '✅ marked', role: 'axis 1+2+3 swap (full SoM)' },
  { name: 'P-prompt', text: 'AXTree', prompt: 'SoM-prompt', image: '❌', role: 'diamond completion (你的设计 ⭐)', highlight: true, special: true },
  { name: 'Vision', text: '(empty)', prompt: 'vision-prompt', image: '✅ raw', role: 'image-only path' },
];

const FIGURES = [
  {
    id: 'fig1ab',
    file: 'fig1ab_cascade_diamond.png',
    title: '3-Axis Cascade Diamond (理论框架)',
    section: 'Theory',
    point: '3-axis ablation cube 的 2D 投影 (image axis off). 4 corner = DOM/P-text/P-SoM/P-prompt. Edge 连接只差一个 axis 的 mode.',
    quote: '"这是 paper Section 5 mechanism 论证的 backbone."',
  },
  {
    id: 'fig0c_oracle',
    file: 'fig0c_drop_one_oracle.png',
    title: 'Drop-one Oracle Lift',
    section: 'Outcome (a) drop-one',
    point: 'Phantom-SoM 在 reddit drop-one 3.33pp (vs SoM 1.90pp) — Phantom unique 解 SoM 解不了的 task. 这是 4-fold drop-in (d) 性质的核心证据.',
    quote: '"Phantom 不是 SoM 的子集 — 是 complementary routing arm."',
  },
  {
    id: 'fig0c_bars',
    file: 'fig0c_phantom_lift_bars.png',
    title: 'Phantom Lift Bars',
    section: 'Outcome (a) drop-one (clean view)',
    point: 'Drop-one oracle 的 cleaner bar chart 视图. 红色 bar 高度 = mode 的 unique contribution.',
  },
  {
    id: 'fig0d',
    file: 'fig0d_taskpool_jaccard.png',
    title: 'Task-pool Jaccard Heatmap',
    section: 'Outcome (b) complementarity',
    point: 'P-text ∩ P-SoM Jaccard 0.500 — 一半 success task 不重叠. P-SoM 跟 SoM Jaccard < 0.7. 反驳 "Phantom 是 SoM 的子集" 假说.',
    quote: '"Phantom finding 不是 dispatch noise — task pool 真不一样."',
  },
  {
    id: 'fig0e',
    file: 'fig0e_category_mode_heatmap.png',
    title: 'Category × Mode Heatmap',
    section: 'Outcome (c) site-modulated',
    point: 'Phantom-SoM 在 text-dominated category (reddit) 强, SoM 在 visual category (cls listing browse) 强. Site-modulated dominance.',
  },
  {
    id: 'fig0f',
    file: 'fig0f_overlap_stacked_bar.png',
    title: 'Mode Overlap Stacked Bar',
    section: 'Outcome (d) Phantom unique',
    point: 'Phantom unique success 数量 (~8 task on reddit) — 这些 task 4-mode oracle 拿不到, 只有 Phantom 能解.',
  },
  {
    id: 'fig0g',
    file: 'fig0g_routing_auroc_heatmap.png',
    title: 'Routing Signal AUROC Heatmap',
    section: '4-fold drop-in (c) AUROC',
    point: '5-mode 全 overall_usable=True. red P-text verbalized 0.793 是 5-mode 最高 (超 baseline 0.766). Phantom 作为 routing arm signal 不输 baseline.',
    quote: '"4-fold drop-in property (c) — routing infrastructure cost-free."',
  },
  {
    id: 'fig1c',
    file: 'fig1c_strategy_gradient.png',
    title: 'Strategy Gradient Across Modes',
    section: 'Macro mechanism',
    point: 'reddit search-loop%: DOM 51.9% → P-SoM 35.7% → SoM 31.4%. 单调下降跟 [SOM_MARKS] 暴露程度一致. Macro action distribution shift, 不是 SR 副产物.',
    quote: '"Macro evidence — 反驳 Phantom 优势是 dispatch bug 假象."',
  },
  {
    id: 'fig2_b0',
    file: 'fig2_micro_divergence_heatmap_B0.png',
    title: 'Micro Step-Divergence Heatmap (B0)',
    section: 'Micro mechanism',
    point: 'DOM vs P-SoM 轨迹首次 diverge 的 step. Median = step 0 — 一开始就走不同路, 不是后期纠错.',
  },
  {
    id: 'fig2_b1',
    file: 'fig2_micro_divergence_heatmap_B1.png',
    title: 'Micro Step-Divergence Heatmap (B1)',
    section: 'Cross-capability micro',
    point: 'B1 (4B) micro divergence pattern 跟 B0 (235B) 类似 — capability tier 不改 mechanism, 只改 absolute SR.',
  },
  {
    id: 'fig3a',
    file: 'fig3a_token_cost_intra_baseline.png',
    title: 'Token Cost Intra-Baseline',
    section: '4-fold drop-in (a) cost',
    point: 'Phantom-SoM 的 token cost 跟 DOM 几乎一样 (red 3437 vs 3661 ±7%). 因为 [SOM_MARKS] 是 AXTree regex filter, 不需额外 model call.',
    quote: '"Phantom 配 DOM-cost — (a) 性质 ✅"',
  },
  {
    id: 'fig3d',
    file: 'fig3d_cost_sr_frontier.png',
    title: 'Cost-SR Pareto Frontier',
    section: 'Efficiency',
    point: 'Pareto frontier 上 Phantom-SoM 在 reddit 是 efficient corner — DOM-cost 等级 + SoM-之上 SR. 配 Vision 高 cost 高 SR 形成完整 cascade.',
  },
  {
    id: 'fig3_carbon',
    file: 'fig3_regional_carbon.png',
    title: 'Regional Carbon Footprint',
    section: 'Sustainability bonus',
    point: 'Cost methodology 用 electricity-equivalent ($0.12/kWh). B0 API ~$0.04/ep vs B1 local ~$0.0004/ep, real ratio ~100×. 适合 NeurIPS green AI / ICLR sustainability framing.',
  },
  {
    id: 'fig_capability',
    file: 'fig_capability_b0_b1.png',
    title: 'Cross-Capability (B0 vs B1)',
    section: 'Generalization',
    point: 'B0 (235B API) vs B1 (4B local). Phantom-SoM 在 B1 上 SR 也比 DOM 高. Mechanism 跨 capability tier 一致 — Section 7 generalization claim.',
  },
];

const KEY_NUMBERS = [
  { label: 'red P-SoM SR', value: '13.81%', context: '> SoM 10.48% > DOM 9.05%' },
  { label: 'red P-SoM drop-one', value: '3.33pp', context: 'vs SoM 1.90pp (Phantom unique)' },
  { label: 'cls SoM SR', value: '21.37%', context: '> P-SoM (image-rich site SoM dominant)' },
  { label: 'cls latency', value: 'SoM 74s vs P-SoM 18.2s', context: '4× faster' },
  { label: 'token cost ratio', value: '~7%', context: 'P-SoM vs DOM (red 3437/3661)' },
  { label: 'cost ratio B0/B1', value: '~100×', context: '$0.04/ep vs $0.0004/ep' },
  { label: 'drop-one range', value: '1.7-3.3pp', context: 'cls 2.56 / red 3.33' },
  { label: 'Jaccard P-text/P-SoM', value: '0.500', context: '< 0.7 sentinel' },
];

const REVIEWER_QA = [
  {
    q: 'Phantom-SoM 是 prior work 已有的 trick 吗?',
    a: '没有. 之前 SoM-style paper (Yang 2023 Set-of-Mark, etc.) 没有 isolate text-payload axis from image axis. 我们的 3-axis ablation framework + diamond completion + 4-fold drop-in characterization 是 paper-original.',
  },
  {
    q: '如果 framework bugs 修了, Phantom 优势会变吗?',
    a: '估算 +2-5pp 整体 SR 提升 (Tier 10 estimate 5.5% off-target → locator-route lift), 但 cross-mode delta robust. Pilot wave-2 (T=0 修复, dispatch bug 仍在) 已经 Δ=0pp on N=60 ep matched-subset SR.',
  },
  {
    q: 'B1 数据没跑完, paper 写得动吗?',
    a: 'B1 cls 现 85% (198/234), 1-2 周内补齐. B0 paper-grade clean (cls + reddit 5-mode 全) 已经够 paper Section 4 主线. B1 是 Section 7 cross-capability evidence, 可分阶段交.',
  },
  {
    q: 'VWA bug 这个发现新颖吗?',
    a: 'Tier 3 Gemini DR 综述显示同领域 paper 普遍 acknowledge silent-failure noise (cite 5-category taxonomy), 但没有 paper 系统 catalog 37 entries + ship fix patches + verify via Playwright replay. 我们的 audit 是 paper-grade rigor 的最强级别.',
  },
  {
    q: '为什么 5-mode 不是 8-mode (2³ cube)?',
    a: '8-mode 里 4 个是 deliberate-ID-mismatched mode (e.g. AXTree text + SoM prompt 但 SoM 用 [mark N] AXTree 用 accessibility ID) — confound prompt effect 跟 parsing-confusion effect, 不是 clean ablation.',
  },
  {
    q: '跟 prior 4-mode VWA paper (e.g. WebArena) 比?',
    a: '我们 5-mode + diamond P-prompt = 6-mode 是已知最系统化 ablation. WebArena 原 paper 只 3-mode, Set-of-Mark paper 只对比 SoM vs Vision.',
  },
];

const ANTI_PATTERNS = [
  { tag: '⚠️ 过早提 bug', detail: '不要开场就说"我发现 framework 有 bug" — 学长第一印象会变成"你的实验数据有问题"' },
  { tag: '⚠️ Overclaim', detail: '不说 "Phantom 总是 win"; 用 site-modulated framing — cls SoM 主导, red Phantom 主导' },
  { tag: '⚠️ Diamond 必须强调是你的设计', detail: 'P-prompt corner 是你加的 factorial completion. paper_planning N1 标注 — 反 overengineering challenge' },
  { tag: '⚠️ 数字 cite 错', detail: '6 个 key numbers 必背 (right column) — 学长 challenge 数字时不结巴' },
  { tag: '⚠️ VWA bug 措辞', detail: '说 "VWA + P79 共有 37 个 scaffold issues", 不说 "framework 全是 bug"' },
  { tag: '⚠️ bug 抢戏', detail: 'main 5 min 讲 Phantom, bug 1 min 提及, deep-dive 留 paper appendix 决策时' },
];

const ASKS = [
  {
    title: 'Ask #1 — RunPod 经费 ~$150-200',
    body: 'DGX GPU 争抢严重 (B1 234 ep 跑 20+ 小时). UCL Myriad 物理级 blocked (firewall drop Tailscale). RunPod 4090 dedicated $0.6/h, 14-cell paper-grade rerun 估 ~$110 + buffer ~$200. 申请课题经费走 RunPod.',
    icon: DollarSign,
  },
  {
    title: 'Ask #2 — Paper scope: Single vs Split',
    body: 'Phantom 主线 1 paper + appendix VWA bugs 披露 vs 拆成 (Phantom paper + VWA bug audit short paper). Bug audit 37 entries + Tier 1-5 + Tier 10 probe 可独立 short paper / workshop.',
    icon: GitBranch,
  },
  {
    title: 'Ask #3 — Section 4 framework + scope',
    body: '4-dimension framework (Outcome / Macro / Micro / Efficiency) + Section 5 site × axis × LLM-mechanism C 框架 — 这个 framing OK 吗? Section 7 cross-capability + Section 8 sustainability 加成.',
    icon: Layers,
  },
];

function HeroProperty({ prop }) {
  const Icon = prop.icon;
  const colorMap = {
    green: 'border-emerald-300 bg-emerald-50 text-emerald-900',
    amber: 'border-amber-300 bg-amber-50 text-amber-900',
    sky: 'border-sky-300 bg-sky-50 text-sky-900',
    violet: 'border-violet-300 bg-violet-50 text-violet-900',
  };
  return (
    <div className={`rounded-xl border-2 p-5 ${colorMap[prop.color]} flex flex-col gap-2`}>
      <div className="flex items-center gap-2">
        <Icon size={20} />
        <span className="font-bold text-lg">{prop.label}</span>
      </div>
      <p className="text-sm opacity-80">{prop.detail}</p>
    </div>
  );
}

function FigureCard({ fig, idx }) {
  const [expanded, setExpanded] = useState(idx < 3);
  return (
    <div className="border border-slate-200 rounded-xl overflow-hidden bg-white shadow-sm hover:shadow-md transition-shadow">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full px-5 py-3 flex items-center justify-between bg-gradient-to-r from-slate-50 to-white hover:from-slate-100"
      >
        <div className="flex items-center gap-3 text-left">
          <span className="text-xs font-mono px-2 py-0.5 bg-indigo-100 text-indigo-700 rounded">
            {fig.section}
          </span>
          <span className="font-semibold text-slate-800">{fig.title}</span>
        </div>
        {expanded ? <ChevronDown size={18} /> : <ChevronRight size={18} />}
      </button>
      {expanded && (
        <div className="px-5 py-4 border-t border-slate-100">
          <div className="rounded-lg overflow-hidden border border-slate-200 bg-slate-50 mb-3">
            {/* relative path so file:// (open dist/index.html directly) + npm run preview both work */}
            <img src={`${import.meta.env.BASE_URL}figures/${fig.file}`} alt={fig.title} className="w-full" loading="lazy" />
          </div>
          <p className="text-sm text-slate-700 leading-relaxed">{fig.point}</p>
          {fig.quote && (
            <div className="mt-3 flex gap-2 items-start text-sm text-indigo-700 bg-indigo-50 rounded-lg p-3 border border-indigo-100">
              <Quote size={14} className="mt-0.5 flex-shrink-0" />
              <span className="italic">{fig.quote}</span>
            </div>
          )}
          <p className="mt-2 text-xs text-slate-400 font-mono">/figures/{fig.file}</p>
        </div>
      )}
    </div>
  );
}

export default function PhantomAdvisorBrief() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-indigo-50">
      <header className="sticky top-0 bg-white/90 backdrop-blur border-b border-slate-200 z-10 px-6 py-4">
        <div className="max-w-6xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Sparkles className="text-indigo-600" size={24} />
            <div>
              <h1 className="font-bold text-lg text-slate-900">Phantom-SoM 学长会议 brief</h1>
              <p className="text-xs text-slate-500">2026-04-30 · 30-45 min · 14 figures + 6 reviewer Q&A + 3 asks</p>
            </div>
          </div>
          <div className="flex gap-2 text-xs">
            <span className="px-2 py-1 bg-emerald-100 text-emerald-700 rounded">Phase A code complete</span>
            <span className="px-2 py-1 bg-amber-100 text-amber-700 rounded">Pilot wave-2 PASS</span>
          </div>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-6 py-8 space-y-12">
        {/* === A. Hero === */}
        <section>
          <h2 className="text-3xl font-bold text-slate-900 mb-2">
            Phantom-SoM 是 hidden 4th routing arm
          </h2>
          <p className="text-lg text-slate-600 mb-6">4-fold drop-in property — paper Section 1 hook</p>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
            {HERO_PROPS.map((p, i) => (
              <HeroProperty key={i} prop={p} />
            ))}
          </div>
          <blockquote className="border-l-4 border-indigo-400 bg-indigo-50/50 px-5 py-4 rounded-r-lg italic text-slate-700">
            "Phantom-SoM identifies a hidden text-only routing arm in SoM-style web agents that
            achieves DOM-level cost and ~50% lower latency while contributing 1.7-3.3pp drop-one
            oracle value. The arm is created by skipping the marked-image draw and image-token
            inference path — no model retraining, no prompt change, no infrastructure overhead."
          </blockquote>
        </section>

        {/* === B. Theory: 6-mode ablation === */}
        <section>
          <div className="flex items-center gap-3 mb-4">
            <Layers className="text-indigo-600" size={28} />
            <h2 className="text-2xl font-bold text-slate-900">3-Axis Ablation Cube + Diamond Completion</h2>
          </div>
          <p className="text-slate-700 mb-4 leading-relaxed">
            Standard VWA 3-mode (DOM/SoM/Vision) 同时换 3 axis (text payload × system prompt × image),
            没法 attribute 哪个 axis 起作用. 我们设计 3 个 phantom mode 做 controlled mismatch — 每个
            phantom 沿单一 axis 偏移 baseline, isolating axis 效应. 最关键: <strong>P-prompt corner</strong>{' '}
            是我加的 diamond completion, 让原本 L-shaped 的 ablation 闭合成 2×2 factorial design.
          </p>
          <div className="overflow-x-auto rounded-lg border border-slate-200 bg-white">
            <table className="w-full text-sm">
              <thead className="bg-slate-50 border-b border-slate-200">
                <tr>
                  <th className="px-4 py-2 text-left font-semibold text-slate-700">Mode</th>
                  <th className="px-4 py-2 text-left font-semibold text-slate-700">Text payload</th>
                  <th className="px-4 py-2 text-left font-semibold text-slate-700">Prompt prior</th>
                  <th className="px-4 py-2 text-left font-semibold text-slate-700">Image</th>
                  <th className="px-4 py-2 text-left font-semibold text-slate-700">Role</th>
                </tr>
              </thead>
              <tbody>
                {SIX_MODES.map((m, i) => (
                  <tr
                    key={i}
                    className={`border-b border-slate-100 ${
                      m.special ? 'bg-violet-50' : m.highlight ? 'bg-indigo-50/40' : 'bg-white'
                    }`}
                  >
                    <td className={`px-4 py-2 font-semibold ${m.special ? 'text-violet-700' : 'text-slate-900'}`}>
                      {m.name}
                    </td>
                    <td className="px-4 py-2 font-mono text-xs text-slate-700">{m.text}</td>
                    <td className="px-4 py-2 font-mono text-xs text-slate-700">{m.prompt}</td>
                    <td className="px-4 py-2 text-center">{m.image}</td>
                    <td className={`px-4 py-2 text-xs ${m.special ? 'font-semibold text-violet-700' : 'text-slate-600'}`}>
                      {m.role}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="mt-4 p-4 bg-violet-50 border border-violet-200 rounded-lg">
            <p className="text-sm text-violet-900">
              <strong className="font-semibold">Diamond 价值</strong>: 没有 P-prompt, ablation 是 L-shape (DOM →
              P-text → P-SoM 三角), 只能测 prompt 在 [SOM_MARKS] text 上的效应. 加 P-prompt 让 4-corner factorial
              design 闭合 — prompt × text 的 interaction 才能 separately quantify. 这是 paper Section 5
              mechanism 论证的<strong> 结构性必要</strong>, 不只是"多跑一组实验". (paper_planning §2 line 47-91 + line 437)
            </p>
          </div>
        </section>

        {/* === C. Empirical figures === */}
        <section>
          <div className="flex items-center gap-3 mb-4">
            <TrendingUp className="text-indigo-600" size={28} />
            <h2 className="text-2xl font-bold text-slate-900">Empirical Walk-through (14 figures)</h2>
          </div>
          <p className="text-slate-600 mb-6">
            按 paper Section 5 mechanism 论证 flow 排序: Theory → Outcome → Macro → Micro → Efficiency → Cross-capability.
            Click 每张卡片展开图 + caption + 关键 quote.
          </p>
          <div className="space-y-3">
            {FIGURES.map((fig, idx) => (
              <FigureCard key={fig.id} fig={fig} idx={idx} />
            ))}
          </div>
        </section>

        {/* === D. Caveats: VWA bugs === */}
        <section>
          <div className="flex items-center gap-3 mb-4">
            <AlertTriangle className="text-amber-600" size={28} />
            <h2 className="text-2xl font-bold text-slate-900">Caveats — VWA Framework Bugs (今日发现)</h2>
          </div>
          <p className="text-slate-700 mb-4 leading-relaxed">
            今天系统 audit 了 VWA framework + P79 wrapper, 找到 <strong>37 个 scaffold-level bugs</strong>
            (Tier 1-5 audit + Tier 10 dispatch-effective-target probe + 4 verification probes), 已 ship Phase A
            4-cluster patch (commit <code className="text-xs bg-slate-100 px-1 rounded">3c15cd7</code>).
          </p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="bg-rose-50 border border-rose-200 rounded-lg p-4">
              <h3 className="font-semibold text-rose-900 mb-2 flex items-center gap-2">
                <XCircle size={18} /> Bugs 影响 absolute SR
              </h3>
              <p className="text-sm text-rose-800">
                ~5-10% inflation/deflation. 最严重 B-33 family (94.4% off-target on failed clicks) 是 AXTree
                element_id mapping 到 child element 而非 actionable parent.
              </p>
            </div>
            <div className="bg-emerald-50 border border-emerald-200 rounded-lg p-4">
              <h3 className="font-semibold text-emerald-900 mb-2 flex items-center gap-2">
                <CheckCircle2 size={18} /> Cross-mode 比较仍 robust
              </h3>
              <p className="text-sm text-emerald-800">
                4 个 text-bearing modes (DOM/SoM/P-text/P-SoM) 共享同样 dispatch contamination — symmetric noise
                cancels. Phantom finding 不受影响 (4 reasons: symmetric / Vision counter-evidence / pilot Δ=0pp
                / architectural design).
              </p>
            </div>
          </div>
          <div className="mt-4 flex flex-wrap gap-2">
            <a
              href="#"
              className="text-xs px-3 py-1 bg-slate-100 hover:bg-slate-200 rounded inline-flex items-center gap-1 text-slate-700"
            >
              <ExternalLink size={12} /> docs/reference/VWA_FRAMEWORK_BUGS_AND_PHASE_A_FIXES.md
            </a>
            <a
              href="#"
              className="text-xs px-3 py-1 bg-slate-100 hover:bg-slate-200 rounded inline-flex items-center gap-1 text-slate-700"
            >
              <ExternalLink size={12} /> docs/checkpoints/master_bug_catalog.md (37 entries)
            </a>
          </div>
        </section>

        {/* === E. Asks === */}
        <section>
          <div className="flex items-center gap-3 mb-4">
            <DollarSign className="text-emerald-600" size={28} />
            <h2 className="text-2xl font-bold text-slate-900">Decision Asks</h2>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {ASKS.map((ask, i) => {
              const Icon = ask.icon;
              return (
                <div key={i} className="border-2 border-emerald-200 bg-emerald-50/50 rounded-xl p-5">
                  <div className="flex items-start gap-3 mb-2">
                    <Icon className="text-emerald-700 flex-shrink-0 mt-1" size={20} />
                    <h3 className="font-semibold text-emerald-900">{ask.title}</h3>
                  </div>
                  <p className="text-sm text-emerald-800 leading-relaxed">{ask.body}</p>
                </div>
              );
            })}
          </div>
        </section>

        {/* === F. Reviewer Q&A === */}
        <section>
          <div className="flex items-center gap-3 mb-4">
            <Quote className="text-indigo-600" size={28} />
            <h2 className="text-2xl font-bold text-slate-900">Reviewer Pushback Prep (6 个 Q&A)</h2>
          </div>
          <div className="space-y-3">
            {REVIEWER_QA.map((qa, i) => (
              <div key={i} className="border border-slate-200 rounded-lg bg-white overflow-hidden">
                <div className="px-4 py-2 bg-amber-50 border-b border-amber-100 text-sm font-semibold text-amber-900 flex items-start gap-2">
                  <span className="text-amber-600 font-mono text-xs">Q{i + 1}</span>
                  <span>{qa.q}</span>
                </div>
                <div className="px-4 py-3 text-sm text-slate-700 leading-relaxed">{qa.a}</div>
              </div>
            ))}
          </div>
        </section>

        {/* === G. Anti-patterns === */}
        <section>
          <div className="flex items-center gap-3 mb-4">
            <AlertTriangle className="text-rose-600" size={28} />
            <h2 className="text-2xl font-bold text-slate-900">Anti-patterns 不要犯的错</h2>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {ANTI_PATTERNS.map((ap, i) => (
              <div key={i} className="border border-rose-200 bg-rose-50/30 rounded-lg p-4">
                <p className="font-semibold text-rose-900 mb-1">{ap.tag}</p>
                <p className="text-sm text-slate-700">{ap.detail}</p>
              </div>
            ))}
          </div>
        </section>

        {/* === H. Cheat sheet — key numbers === */}
        <section>
          <div className="flex items-center gap-3 mb-4">
            <Cpu className="text-slate-600" size={28} />
            <h2 className="text-2xl font-bold text-slate-900">Cheat Sheet — Key Numbers (打印带去会议)</h2>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {KEY_NUMBERS.map((kn, i) => (
              <div key={i} className="border border-slate-200 bg-white rounded-lg p-3">
                <p className="text-xs text-slate-500 mb-1">{kn.label}</p>
                <p className="font-mono font-bold text-lg text-slate-900">{kn.value}</p>
                <p className="text-xs text-slate-500 mt-1">{kn.context}</p>
              </div>
            ))}
          </div>
        </section>

        {/* === Footer === */}
        <footer className="border-t border-slate-200 pt-6 mt-12 text-sm text-slate-500">
          <p>
            <strong>References</strong>: docs/checkpoints/paper_planning.md (1255 行 strategy notebook) ·
            docs/reference/VWA_FRAMEWORK_BUGS_AND_PHASE_A_FIXES.md ·
            docs/reference/MYRIAD_SMOKE_REPORT.md ·
            docs/reference/PHANTOM_SOM_ADVISOR_MEETING_BRIEF.md (text version)
          </p>
          <p className="mt-2">
            <strong>State</strong>: master @ commit 578805b · pilot wave-2 PASS · Phase A code complete · 14-cell rerun pending RunPod approval.
          </p>
        </footer>
      </main>
    </div>
  );
}
