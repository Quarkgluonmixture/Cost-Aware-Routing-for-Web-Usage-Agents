/**
 * Weekly Report Data — 2026-04-16
 *
 * To create a new week's report:
 * 1. Copy this file to weekly_X_XX.js
 * 2. Update all data sections below
 * 3. Change the import in App.jsx: import data from './data/weekly_X_XX.js'
 * 4. Run: npm run build
 */

const data = {
  // --- Header ---
  header: {
    date: '2026-04-16',
    title: 'B1 Tri-Mode Results + Routing Framework + B0 Prepared',
    phase: 'Phase 1 Representation Screening',
    status: 'Classifieds B1 Complete \u00b7 B0 Prepared',
    models: 'Qwen3-VL-4B / 235B',
  },

  // --- Data Status Disclaimer ---
  disclaimer: {
    title: 'Data Status Disclaimer (Preliminary)',
    summary: 'All SRs and signal evaluations this week are preliminary.',
    items: [
      'B1 Classifieds data suffers from a reference image transmission bug in DOM mode (Triple bug in transfer/encoding/labeling \u00a733/\u00a734/\u00a736 fixed, but B1 data was collected prior to the fix).',
      'B1 contains <code>state_change</code> false negatives/positives (\u00a768 fixed, Reddit rerunning from scratch).',
      'B0 exhibits a ~20.1% <code>parse_error</code> rate (\u00a767 root fix implemented, awaiting API quota to rerun).',
    ],
    footer: 'Conclusions are subject to change pending final data updates.',
  },

  // --- Section 1: SR Data ---
  srData: [
    { name: 'DOM', raw: 8.97, adjusted: 0.85, cost: 0.074 },
    { name: 'SoM', raw: 20.51, adjusted: 16.24, cost: 0.077 },
    { name: 'Vision', raw: 12.39, adjusted: 8.12, cost: 0.029 },
  ],

  significance: [
    { pair: 'SoM vs DOM', pValue: 'p=7.4e-6 (Significant)' },
    { pair: 'SoM vs Vision', pValue: 'p=0.0013 (Significant)' },
    { pair: 'Vision vs DOM', pValue: 'p=0.0002 (Significant)' },
  ],

  costNote: 'Vision cost is only <strong>39%</strong> of DOM (Wilcoxon p<1e-25)',

  domBugNote: 'DOM mode suffered from a triple bug (\u00a733/\u00a734/\u00a736) preventing the use of reference images entirely during B1 data collection. This is fixed in subsequent runs, meaning current DOM SR might be artificially low.',

  // --- Section 1.1-1.2: False Positives ---
  falsePositives: {
    visual: {
      title: '1.1 Visual False Positive (\u00a720)',
      description: 'DOM mode lacks vision, yet <strong>19 out of 21 "successes" were lucky hits on visual tasks</strong> (e.g., asked to find a "blue kayak", DOM cannot see colors, but the first search result happened to be the target item).',
      detail: 'In Classifieds, <strong>67% of tasks are visual (162/234)</strong>, severely inflating DOM\'s raw SR. <strong>Post-adjustment, DOM\'s true SR is only 0.85% (2/234)</strong>\u2014pure text representation is effectively useless on visually dense sites.',
      b0Note: '<strong>B0 (235B) Complexity (\u00a756):</strong> 35 Successes = 7 na_fp + 10 visual_fp + 17 new blind spots (pending manual verification) + 1 true success. Adjusted SR 7.7% ~ 0.4%. The LLM shows emergent capabilities (pagination, price filters, tab_focus) absent in 4B. A finer-grained FP taxonomy is required.',
    },
    naTask: {
      title: '1.2 N/A Task ua_match False Positive (\u00a727)',
      description: 'VWA has 10 N/A reference tasks (labeled "unachievable"). The <code>ua_match</code> evaluator using GPT-4o-mini evaluated these, and <strong>all 10 were falsely judged as successes</strong>:',
      types: [
        '<strong>Type A:</strong> Agent yields empty answer (truncated by max_steps) \u2192 Judged as "implicit admission of N/A" \u2192 1.0',
        '<strong>Type B:</strong> Agent submits wrong answer \u2192 "even if implicitly" in prompt is too lenient \u2192 Still judged as "same".',
      ],
      rootCauses: '<strong>Dual Root Causes:</strong> (1) ua_match prompt pre-assumes the agent is answering N/A; (2) agent Rule 4 dictates "NEVER give up", leaving no exit for N/A, guaranteeing a truncation cycle.',
      impact: '* Impact: DOM 10/10, SoM 4/4 all FP. Since all 10 are visual tasks, they are covered by Visual FP filtering and don\'t further decrease adjusted SR.',
    },
  },

  // --- Section 1.3: Oracle ---
  oracle: {
    ceiling: '19.66%',
    headroom: '+3.42pp',
    visionOnly: '7 Tasks',
    domVisionOverlap: '0',
    conclusion: '<strong>SoM \u2194 Vision routing</strong> is the primary direction (headroom 3.42pp). DOM is excluded from routing consideration (adjusted SR 0.85%, only 1 exclusive success). Oracle Distribution: SoM ~25 / Vision ~19 / DOM 2 / SoM+Vision (excluding DOM) = 12.',
  },

  // --- Section 2: Behavioral Findings ---
  findings: {
    mirage: {
      title: '2.1 DOM vs SoM Gap Root Cause: Mirage Effect (\u00a718/\u00a723)',
      intro: 'The initial hypothesis was "SoM text is more compact, DOM AXTree is too long leading to information flooding." \u00a723 corrects this: <strong>The Mirage Effect is the primary cause.</strong> Recent literature (Asadi et al., arXiv:2603.21687, Stanford 2026-04) found VLMs confidently describe visual features even without images, achieving 70-80% of their vision-enabled accuracy.',
      comparisons: [
        { paper: 'Blind accuracy is 70-80% of vision accuracy', observation: '90.5% of DOM "successes" are visual lucky hits' },
        { paper: 'Mirage-mode > Guess-mode', observation: 'SoM degradation scenarios (missing images) actually boost SR' },
        { paper: 'Message format triggers reasoning path switch', observation: 'SoM prompt implies screenshot existence \u2192 Activates hidden text-visual associations' },
      ],
      explanatoryPower: 'The textual information volume between DOM and SoM is similar (flat SoM marks vs hierarchical AXTree), but merely mentioning the "screenshot" in the SoM prompt triggers a qualitative shift in reasoning. It\'s not that "seeing the image is better", but rather "being told there is an image is better."',
    },
    cognitiveGap: {
      title: '2.2 The Cognitive-Execution Gap (\u00a731)',
      caseStudy: 'Case study (Task 58: Find oldest item after City filter):',
      domSom: 'The <code>thought</code> explicitly mentions pagination ("I should go to the next page") and filtering intentions, but <strong>cannot execute</strong>\u2014it doesn\'t know which element to click or where the control is.',
      vision: 'Successfully executes by visually locating the pagination button and clicking through to page 3.',
      broader: 'Failures in text modes aren\'t always "not knowing what to do" (cognitive), but rather "knowing what to do but failing to act" (execution). Spatial information bridges this gap. Broadly: Agents rarely page/sort, rarely scroll up (DOM 3.7%, SoM 11.2%). Visuals make controls salient (SoM scroll up 3\u00d7 DOM).',
    },
    selfCorrection: {
      title: '2.3 Zero Self-Correction (\u00a732)',
      description: 'Vision task analysis reveals <strong>systematic coordinate shifts</strong> (e.g., misclicking Arts+Crafts instead of Books).',
      keyPoint: 'Crucially, after a misclick, the agent retries with the <strong>exact same coordinates and confidence</strong>. It does not micro-adjust.',
      implication: 'This implies that auto-scroll/retry fault-tolerance modules have zero impact on Vision mode SR\u2014there is no "retry opportunity" to salvage.',
    },
  },

  // --- Section 3: Routing ---
  signalEvalData: [
    { type: 'Token-level', signal: 'mean/min logprob, margin, entropy (6 metrics)', auroc: '0.497', conclusion: 'Near random. 4B is overconfident. Excluded from Phase 2.' },
    { type: 'Behavioral', signal: 'action diversity (unique action drop rate)', auroc: '0.74', conclusion: 'Consistent across modes. Strongest free signal.' },
    { type: 'Behavioral', signal: 'page_unchanged_streak, URL revisit', auroc: 'Impl.', conclusion: 'R2D2 Lit: 50% error reduction.' },
    { type: 'Verbalized', signal: 'confidence: 0.0-1.0 self-reported in prompt', auroc: '0.69', conclusion: 'Usable for SoM+Vision. Orthogonal to behavioral (low \u03c1).' },
  ],

  signalNote: '* <strong>Token-level signal failure is heavily supported by literature:</strong> 8 surveys (\u00a715) consistently indicate that raw logprobs in small VLMs are overconfident and unreliable before calibration. Most practical route: <strong>Free behavioral signals trigger mode upgrades</strong> (modifying existing cycle detection from "trigger retry" to "trigger mode switch").',

  routingLitData: [
    { signal: 'page_unchanged_streak / no_progress_streak', status: 'Implemented', reference: 'state_change.py' },
    { signal: 'URL Revisit Detection', status: 'In Analysis', reference: 'R2D2 (50% error reduction)' },
    { signal: 'Action Diversity', status: 'In Analysis', reference: 'WebGym, TGPO (AUROC=0.74)' },
    { signal: 'PRM Verifier (7B)', status: 'Discarded', reference: 'WebArbiter +9.1pp, but side-running 7B next to 4B is uneconomical.' },
    { signal: 'Learned Router', status: 'Discarded', reference: 'FrugalGPT 98% cost cut, but requires training data.' },
  ],

  ablationData: [
    { group: 'A: DOM', prompt: 'DOM', text: 'AXTree', image: 'None', variable: 'Baseline (B1)', bg: 'bg-slate-50' },
    { group: 'B: Phantom-SoM', prompt: 'SoM', text: 'SoM marks', image: 'None', variable: '\u2460+\u2461+\u2462 Mixed', bg: 'bg-indigo-50 font-medium' },
    { group: 'C: Full SoM', prompt: 'SoM', text: 'SoM marks', image: 'Yes', variable: 'Pure Image (B1)', bg: 'bg-slate-50' },
    { group: 'D: DOM-text+SoM-prompt', prompt: 'SoM', text: 'AXTree', image: 'None', variable: 'Pure \u2460 Prompt', bg: 'bg-emerald-50 font-medium' },
    { group: 'E: SoM-text+DOM-prompt', prompt: 'DOM', text: 'SoM marks', image: 'None', variable: 'Pure \u2461 Format', bg: 'bg-slate-50' },
  ],

  phantomSom: {
    intro: 'If the Mirage Effect holds, a zero-cost intermediate layer exists: <strong>Phantom-SoM</strong> (SoM prompt + SoM marks text, but NO image). Theoretically, it captures most of SoM\'s gains at DOM\'s cost.',
    causalLogic: '<strong>Causal Breakdown Logic:</strong> <code>D - A</code> = prompt effect, <code>E - A</code> = format effect, <code>C - B</code> = image effect, <code>B - D - E + A</code> \u2248 mirage effect.',
    critical: '<strong>Group D is the critical discriminant:</strong> Scaffold Effect literature (Vu & Balloccu 2026\u2014merely mentioning MRI availability explains 70-80% of performance variance) predicts D \u2248 B. If true, the effect stems entirely from the prompt layer.',
    novelty: 'A/C reuses B1 data (zero cost), B/D require 234 tasks each.<br/><strong>P79 Novelty:</strong> First intra-model observation-mode routing study + systematic 4-factor ablation + treating mirage effect as a routing feature, not a bug.',
  },

  // --- Section 4: B0 ---
  b0Observations: [
    { icon: 'Zap', color: 'indigo', title: 'Emergent Capabilities', description: 'Pagination (never seen in B1), price filtering (active use of filter controls), tab_focus (multi-tab comparison). These behaviors are completely absent in 4B, <strong>validating the core premise of capability-aware routing (\u00a79)</strong>.' },
    { icon: 'ListTodo', color: 'orange', title: 'More Complex FP Taxonomy', description: 'Out of 35 successes, only 1 was "definitively true". The rest: 7 na_fp / 10 visual_fp / 17 unverified blind spots. Demands a finer-grained False Positive classification system than 4B.' },
    { icon: 'Bug', color: 'red', title: 'Parse Error Crisis (20.1%)', description: 'Bedrock lacks support for thinking separation, causing 235B reasoning text to mix with JSON, breaking the keyword fallback. <strong>Dual root-fix (Tool Calling + GLM Extraction) implemented</strong>, awaiting API quota.' },
    { icon: 'ArrowRight', color: 'slate', title: 'Scroll Direction Confusion (\u00a767)', description: 'deltaY sign conventions differ across CSS/Win32/macOS. 235B\'s training data mixes all three, leading to unpredictable scrolling. Fixed by implementing <strong>scroll_direction enum semanticization</strong>.' },
  ],

  // --- Section 5: Scaffold Fixes ---
  scaffoldFixes: [
    { id: '\u00a767', issue: 'B0 parse_error (thinking mixed output)', impact: 'B0 SoM 20.1% step degradation', status: 'Fixed, pending rerun' },
    { id: '\u00a768', issue: 'state_change False Negative/Positive', impact: 'Inaccurate action_success in B1', status: 'Fixed, B1 rerunning' },
    { id: '\u00a751', issue: 'select_option unhandled (native <select> unclickable)', impact: 'All B0/B1 publish tasks failed', status: 'Fixed' },
    { id: '\u00a760-62', issue: 'CSS custom dropdowns (Sort by / Reddit sort)', impact: 'Unclickable in DOM/SoM/Vision', status: 'Fixed' },
    { id: '\u00a763', issue: '<think> tag inducing parse_error', impact: 'B0 DOM 16 / Vision 33 tasks', status: 'Fixed' },
    { id: '\u00a757', issue: 'tab_focus loop misclassification', impact: 'Multi-tab tasks killed by cycle detect', status: 'Fixed' },
    { id: '\u00a753', issue: 'confirm dialog blocking', impact: 'All delete tasks failed', status: 'Fixed' },
  ],

  otherInfra: 'B0 API Proxy Agent (\u00a717), VWA Site Reset Mechanism (\u00a738), Verbalized Confidence Extraction (\u00a726), GLM Batch Digest Cross-Mode Enhancement (\u00a765), Cost Model Completion (\u00a735: token classification + 3-stage latency + FLOPs estimation).',

  // --- Section 6: Status ---
  executionStatus: [
    { task: 'B1 Classifieds', status: 'Stub (No Rerun)', note: 'Adjusted SR known: DOM 0.85% / SoM 16.24% / Vision 8.12%', color: 'border-slate-200 bg-slate-50' },
    { task: 'B1 Reddit', status: 'Running', note: 'Running from scratch after \u00a768 fix (DOM ongoing)', color: 'border-blue-200 bg-blue-50 text-blue-900' },
    { task: 'B1 Shopping', status: 'Queued', note: 'Will auto-start after Reddit completes', color: 'border-slate-200 bg-white' },
    { task: 'B0 Classifieds', status: 'Running', note: 'Data cleared. Rerun started with \u00a767+\u00a768 fixes', color: 'border-blue-200 bg-blue-50 text-blue-900' },
  ],

  nextMilestones: [
    'Complete B1 Reddit/Shopping \u2192 Cross-site Aggregation + B0 vs B1 Comparison',
    'Launch Phantom-SoM Ablation Experiments (Groups B & D)',
  ],
};

export default data;
