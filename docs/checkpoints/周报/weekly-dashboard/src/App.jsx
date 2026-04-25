import React from 'react';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer 
} from 'recharts';
import { 
  BarChart2, 
  Lightbulb, 
  AlertTriangle, 
  Activity, 
  Layers, 
  Settings, 
  Calendar,
  Info,
  CheckCircle2,
  XCircle,
  Clock
} from 'lucide-react';

// ==========================================
// 📊 DATA STORE (Externalized per request)
// ==========================================

const METADATA = {
  title: "P79 Weekly Report: Classifieds Full Matrix Completed + Cross-Model Comparison Finalized",
  tags: [
    { label: "Date", value: "2026-04-23" },
    { label: "Phase", value: "Phase 1 Rep. Screening" },
    { label: "Status", value: "VWA Classifieds B0+B1 Done · B0 Reddit WIP" },
    { label: "Models", value: "Qwen3-VL-235B / 4B" }
  ],
  disclaimer: "Data Note: All 6 cells for Classifieds (B0×3 + B1×3) completed and finalized. B1 re-run after fixing viewport ratio bug (§80) and reference image passing; B0 re-run after fixing parse_error (§67). SRs are all adjusted (excluding false positives, denominator 224)."
};

const SUCCESS_RATES_DATA = [
  { mode: 'DOM', B0: 8.48, B1: 4.91, diff: '+3.57pp', rawB0: '19/224', rawB1: '11/224' },
  { mode: 'SoM', B0: 20.98, B1: 13.84, diff: '+7.14pp', rawB0: '47/224', rawB1: '31/224' },
  { mode: 'Vision', B0: 12.05, B1: 7.14, diff: '+4.91pp', rawB0: '27/224', rawB1: '16/224' }
];

const FAILURE_MODES_DATA = [
  { type: 'No-progress loop', dom: '36.8%', som: '36.8%', vision: '58.1%', isVisionMax: true, feature: 'Inaccurate Vision coordinates lead to repeated invalid actions' },
  { type: 'Premature termination', dom: '6.0%', som: '12.0%', vision: '14.1%', isSomMax: true, feature: 'High information density on SoM first screen, model answers without full exploration' },
  { type: 'Answer mismatch', dom: '10.7%', som: '6.4%', vision: '2.1%', isDomMax: true, feature: 'DOM lacks visual info, submitted answers deviate significantly' }
];

const ROUTING_SIGNALS_DATA = [
  { signal: 'Verbalized Confidence', raw: 'verbalized', auroc: '0.769', ci: '[0.704, 0.832]', dom: '0.753', som: '0.755', vision: '0.757', isStrong: true },
  { signal: 'URL Revisit Max', raw: 'url_revisit_max', auroc: '0.767', ci: '[0.718, 0.814]', dom: '0.755', som: '0.727', vision: '0.816', isStrong: true },
  { signal: 'Action Diversity', raw: 'action_diversity', auroc: '0.749', ci: '[0.688, 0.810]', dom: '0.738', som: '0.706', vision: '0.809', isStrong: true },
  { signal: 'Max Repeat Streak', raw: 'max_repeat_streak', auroc: '0.673', ci: '[0.607, 0.735]', dom: '0.761', som: '0.604', vision: '0.744', isStrong: false },
  { signal: 'Token Level (entropy/margin, etc.)', raw: '', auroc: '≈0.50', ci: '—', dom: '—', som: '—', vision: '—', isStrong: false }
];

const ORACLE_ROUTING_DATA = [
  { metric: 'Best Single Mode (SoM)', b0: '20.98%', b1: '13.84%' },
  { metric: 'Oracle Tri-Mode', b0: '29.06%', b1: '18.80%', isBold: true },
  { metric: 'Improvement Room', b0: '+8.55pp', b1: '+5.13pp' }
];

const VENN_B1_DATA = [
  { set: 'SoM Only', count: '19', note: 'Largest exclusive set, SoM is the primary force', isBold: true },
  { set: 'SoM + Vision (Not DOM)', count: '8', note: 'Shared advantage of visual info', isBold: false },
  { set: 'DOM Only', count: '7', note: 'Irreplaceable value of exact element positioning', isBold: true },
  { set: 'Vision Only', count: '5', note: 'Low-cost alternative path', isBold: false },
  { set: 'DOM + SoM (Not Vision)', count: '3', note: '—', isBold: false },
  { set: 'All 3 Modes Success', count: '2', note: 'Simple tasks, minor differences', isBold: false },
  { set: 'DOM + Vision (Not SoM)', count: '0', note: 'DOM and Vision success sets completely non-overlapping', isBold: true }
];

const VENN_B0_DATA = [
  { set: 'SoM Only', count: '27', note: 'Still the largest exclusive set', isBold: true },
  { set: 'Vision Only', count: '15', note: '235B has stronger visual ability, Vision exclusives increased significantly', isBold: true },
  { set: 'All 3 Modes Success', count: '9', note: 'DOM is the cheapest on these intersection tasks ($0.013 vs SoM $0.020)', isBold: false },
  { set: 'DOM + SoM / SoM + Vision', count: '6 each', note: '—', isBold: false },
  { set: 'DOM Only', count: '4', note: '—', isBold: false }
];

const PROGRESS_DATA = [
  { site: 'Classifieds (234)', b0: '✅ 3/3', b1: '✅ 3/3', completed: '6/6', isBold: true },
  { site: 'Reddit (210)', b0: '🔄 dom 137/210', b1: '✅ dom; ⏸ som/vision diff 22+37', completed: '1/6', isBold: false },
  { site: 'Shopping (466)', b0: '✅ dom', b1: '❌', completed: '1/6', isBold: false }
];

const FIXES_DATA = [
  { id: '§80', desc: 'Viewport filter threshold calculation bug', impact: 'DOM/SoM completely re-run' },
  { id: '§81', desc: 'Wikipedia version mismatch causing 404', impact: '35 tasks cleaned and re-run' },
  { id: '§78/83/88', desc: 'False positive detection system refinement', impact: 'SR numbers more accurate' }
];

const NEXT_WEEK_FOCUS = [
  "B0 Reddit dom→som→vision (Automated run)",
  "B1 Reddit som/vision completion (Manual start, diff 22+37)",
  "Reddit Analysis Document",
  "B0 Shopping analysis + som/vision start",
  "WA Full Site (Infrastructure ready)"
];

// ==========================================
// 🧩 COMPONENTS
// ==========================================

const SectionHeader = ({ num, title, icon: Icon }) => (
  <div className="flex items-center gap-4 mb-6">
    <div className="w-12 h-12 rounded-full bg-indigo-100 text-indigo-700 flex items-center justify-center font-bold text-2xl shadow-sm">
      {num}
    </div>
    <h2 className="text-3xl font-bold text-slate-800 flex items-center gap-3">
      <Icon className="w-8 h-8 text-indigo-600" /> 
      {title}
    </h2>
  </div>
);

const AlertBox = ({ type = 'info', children }) => {
  const styles = {
    info: "bg-blue-50 border-blue-200 text-blue-800",
    warning: "bg-amber-50 border-amber-200 text-amber-800",
    success: "bg-emerald-50 border-emerald-200 text-emerald-800"
  };
  const Icon = type === 'warning' ? AlertTriangle : (type === 'success' ? CheckCircle2 : Info);
  
  return (
    <div className={`p-5 rounded-xl border flex gap-4 text-base ${styles[type]}`}>
      <Icon className="w-6 h-6 flex-shrink-0 mt-0.5" />
      <div>{children}</div>
    </div>
  );
};

// ==========================================
// 🚀 MAIN APPLICATION
// ==========================================

export default function Dashboard() {
  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 py-10 px-4 md:px-8 font-sans selection:bg-indigo-200 selection:text-indigo-900">
      <div className="max-w-[1400px] mx-auto space-y-12">
        
        {/* HEADER SECTION */}
        <header className="bg-white rounded-2xl p-8 shadow-sm border border-slate-200 space-y-6">
          <div className="flex flex-wrap gap-3">
            {METADATA.tags.map((tag, idx) => (
              <span key={idx} className="px-4 py-1.5 rounded-full bg-slate-100 text-slate-700 text-sm font-semibold tracking-wide uppercase">
                <span className="text-slate-400 mr-2">{tag.label}:</span>{tag.value}
              </span>
            ))}
          </div>
          <h1 className="text-4xl md:text-5xl font-extrabold text-slate-900 leading-tight">
            {METADATA.title}
          </h1>
          <AlertBox type="warning">
            <strong className="font-bold">Data Note:</strong> {METADATA.disclaimer}
          </AlertBox>
        </header>

        {/* SECTION 1 */}
        <section className="bg-white rounded-2xl p-8 shadow-sm border border-slate-200">
          <SectionHeader num="1" title="B0 vs B1 Complete Results" icon={BarChart2} />
          
          <div className="space-y-8">
            <div>
              <h3 className="text-2xl font-bold text-slate-800 mb-6">1.1 Success Rate</h3>
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-center">
                <div className="h-[300px] w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={SUCCESS_RATES_DATA} margin={{ top: 20, right: 30, left: 0, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
                      <XAxis dataKey="mode" tick={{fontSize: 16, fill: '#475569'}} axisLine={false} tickLine={false} />
                      <YAxis tick={{fontSize: 14, fill: '#475569'}} axisLine={false} tickLine={false} tickFormatter={(val) => `${val}%`} />
                      <Tooltip cursor={{fill: '#f1f5f9'}} contentStyle={{borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)'}} />
                      <Legend wrapperStyle={{paddingTop: '20px'}} />
                      <Bar dataKey="B0" name="B0 (235B)" fill="#4f46e5" radius={[4, 4, 0, 0]} barSize={40} />
                      <Bar dataKey="B1" name="B1 (4B)" fill="#94a3b8" radius={[4, 4, 0, 0]} barSize={40} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                
                <div className="overflow-x-auto rounded-xl border border-slate-200">
                  <table className="w-full text-left text-base">
                    <thead className="bg-slate-50 text-slate-600 font-semibold border-b border-slate-200">
                      <tr>
                        <th className="p-4">Mode</th>
                        <th className="p-4">B0 (235B)</th>
                        <th className="p-4">B1 (4B)</th>
                        <th className="p-4 text-emerald-600">Difference</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100 bg-white">
                      {SUCCESS_RATES_DATA.map((row, idx) => (
                        <tr key={idx} className={row.mode === 'SoM' ? 'bg-indigo-50/50' : ''}>
                          <td className={`p-4 font-medium ${row.mode === 'SoM' ? 'text-indigo-900 font-bold' : 'text-slate-800'}`}>{row.mode}</td>
                          <td className={`p-4 ${row.mode === 'SoM' ? 'font-bold' : ''}`}>{row.B0}% <span className="text-slate-400 text-sm">({row.rawB0})</span></td>
                          <td className={`p-4 ${row.mode === 'SoM' ? 'font-bold' : ''}`}>{row.B1}% <span className="text-slate-400 text-sm">({row.rawB1})</span></td>
                          <td className={`p-4 ${row.mode === 'SoM' ? 'font-bold text-indigo-600' : 'text-emerald-600'}`}>{row.diff}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
              <p className="mt-4 text-lg text-slate-700 bg-slate-50 p-4 rounded-xl">
                B0 dominates B1 across all three modes. The gap in SoM is the largest (+7.14pp), while DOM is the smallest (+3.57pp).
              </p>
            </div>

            <div className="pt-8 border-t border-slate-100">
              <h3 className="text-2xl font-bold text-slate-800 mb-4">1.2 B1 Mode Ranking Correction</h3>
              <div className="bg-white border-2 border-indigo-100 rounded-xl p-6 shadow-sm">
                <p className="text-lg text-slate-700 mb-4">
                  Last week's conclusion was SoM &gt;&gt; Vision &gt;&gt; DOM (three distinct tiers). This week, after fixing the bugs and re-running, the gap between Vision and DOM is no longer significant (p=0.701):
                </p>
                <div className="bg-indigo-50 text-indigo-900 text-2xl font-bold p-6 rounded-xl text-center mb-4">
                  New Ranking: SoM &gt;&gt; Vision ≈ DOM
                </div>
                <p className="text-lg text-slate-700">
                  After fixing two bugs, B1 DOM rose from 0.85% to 4.91%, producing 7 exclusive successful tasks, restoring its routing value.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* SECTION 2 */}
        <section className="bg-white rounded-2xl p-8 shadow-sm border border-slate-200">
          <SectionHeader num="2" title="Unique Value of DOM Mode" icon={Lightbulb} />
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-slate-50 p-6 rounded-xl border border-slate-200">
              <h4 className="text-xl font-bold text-slate-800 mb-3 flex items-center gap-2">
                <span className="w-8 h-8 rounded bg-white shadow-sm flex items-center justify-center text-indigo-600">4B</span>
                B1 DOM Exclusives
              </h4>
              <p className="text-lg text-slate-700 leading-relaxed">
                B1 DOM has <strong>7 exclusive successes</strong> (both SoM and Vision failed): 4 via exact <code className="bg-slate-200 px-1 rounded text-slate-800">element_id</code> click navigation, and 3 via text extraction yielding correct answers. These tasks rely on exact element positioning, where DOM's id mechanism is more reliable than coordinate clicking (Vision) or annotated screenshots (SoM).
              </p>
            </div>
            
            <div className="bg-indigo-50 p-6 rounded-xl border border-indigo-100">
              <h4 className="text-xl font-bold text-indigo-900 mb-3 flex items-center gap-2">
                <span className="w-8 h-8 rounded bg-white shadow-sm flex items-center justify-center text-indigo-600">235B</span>
                B0 DOM Dominance
              </h4>
              <p className="text-lg text-slate-700 leading-relaxed">
                B0 DOM performance is much stronger (8.48% vs 4.91%, <strong>1.73x</strong>). The 235B model exhibits behaviors the 4B model lacks: active pagination (33+ tasks), using price filters (21+ tasks), and multi-tab switching.
              </p>
            </div>
          </div>
          
          <div className="mt-6 bg-emerald-50 border border-emerald-200 text-emerald-900 p-6 rounded-xl text-center text-xl shadow-sm">
            <strong>The value of DOM is positively correlated with model capability.</strong>
          </div>
        </section>

        {/* SECTION 3 */}
        <section className="bg-white rounded-2xl p-8 shadow-sm border border-slate-200">
          <SectionHeader num="3" title="Failure Mode Analysis" icon={AlertTriangle} />
          
          <div className="space-y-10">
            <div>
              <h3 className="text-2xl font-bold text-slate-800 mb-6">3.1 B1 Three-Mode Failure Distribution</h3>
              <div className="overflow-hidden rounded-xl border border-slate-200 shadow-sm">
                <table className="w-full text-left text-base">
                  <thead className="bg-slate-50 text-slate-700 font-semibold border-b border-slate-200">
                    <tr>
                      <th className="p-4 w-1/4">Failure Type</th>
                      <th className="p-4">DOM</th>
                      <th className="p-4">SoM</th>
                      <th className="p-4">Vision</th>
                      <th className="p-4 w-2/5">Feature</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100 bg-white">
                    {FAILURE_MODES_DATA.map((row, idx) => (
                      <tr key={idx} className="hover:bg-slate-50 transition-colors">
                        <td className="p-4 font-medium text-slate-800">{row.type}</td>
                        <td className={`p-4 ${row.type === 'Answer mismatch' ? 'font-bold text-indigo-700 bg-indigo-50/50' : 'text-slate-600'}`}>{row.dom}</td>
                        <td className={`p-4 ${row.type === 'Premature termination' ? 'font-bold text-indigo-700 bg-indigo-50/50' : 'text-slate-600'}`}>{row.som}</td>
                        <td className={`p-4 ${row.type === 'No-progress loop' ? 'font-bold text-indigo-700 bg-indigo-50/50' : 'text-slate-600'}`}>{row.vision}</td>
                        <td className="p-4 text-slate-600 text-sm leading-relaxed">{row.feature}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="mt-4 text-lg text-slate-700 p-4 bg-slate-50 rounded-xl">
                The failure paths of the three modes are distinctly different, indicating that they each have their own pros and cons on different task types. This provides the foundation for routing.
              </p>
            </div>

            <div>
              <h3 className="text-2xl font-bold text-slate-800 mb-4">3.2 B0 vs B1 Behavior Differences</h3>
              <div className="bg-white border border-slate-200 rounded-xl p-6 text-lg text-slate-700 leading-relaxed shadow-sm">
                <p>
                  <strong>235B</strong> attempts to adjust coordinates and retry after a click failure, whereas <strong>4B</strong> repeatedly clicks using the exact same coordinates.
                </p>
                <p className="mt-4">
                  However, <strong>235B exhibits an "overconfidence" issue</strong>: if it sees partial matching results on the first screen, it immediately submits an answer without continuing to paginate or scroll. This is particularly noticeable in <strong>SoM mode</strong>—which has the highest information density on the first screen. Combined with 235B's high confidence, it is the most prone to premature termination.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* SECTION 4 */}
        <section className="bg-white rounded-2xl p-8 shadow-sm border border-slate-200">
          <SectionHeader num="4" title="Routing Signal Evaluation" icon={Activity} />
          
          <p className="text-lg text-slate-700 mb-6">
            AUROC measures the signal's ability to distinguish between success and failure (0.5 is the random baseline). Based on B1 tri-mode 702 episodes (adjusted labels):
          </p>

          <div className="overflow-hidden rounded-xl border border-slate-200 shadow-sm mb-6">
            <table className="w-full text-left text-base">
              <thead className="bg-slate-50 text-slate-700 font-semibold border-b border-slate-200">
                <tr>
                  <th className="p-4">Signal</th>
                  <th className="p-4">AUROC</th>
                  <th className="p-4">95% CI</th>
                  <th className="p-4">DOM</th>
                  <th className="p-4">SoM</th>
                  <th className="p-4">Vision</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 bg-white">
                {ROUTING_SIGNALS_DATA.map((row, idx) => (
                  <tr key={idx} className={row.isStrong ? 'bg-indigo-50/30' : ''}>
                    <td className={`p-4 text-slate-800 ${row.isStrong ? 'font-bold' : ''}`}>
                      {row.signal}
                      {row.raw && <span className="block text-sm text-slate-400 font-normal mt-0.5 font-mono">{row.raw}</span>}
                    </td>
                    <td className={`p-4 ${row.isStrong ? 'font-bold text-indigo-700' : 'text-slate-600'}`}>{row.auroc}</td>
                    <td className="p-4 text-slate-500 font-mono text-sm">{row.ci}</td>
                    <td className="p-4 text-slate-600">{row.dom}</td>
                    <td className="p-4 text-slate-600">{row.som}</td>
                    <td className="p-4 text-slate-600">{row.vision}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="bg-slate-50 rounded-xl p-6 border border-slate-200 space-y-4 text-lg text-slate-700">
            <p>
              <strong>Verbalized Confidence</strong> is the most stable across modes (tri-mode AUROC diff is only 0.004), making it the most suitable unified criterion for cross-mode routing.
            </p>
            <p>
              <strong>Behavioral Signals</strong> (<code className="text-sm">url_revisit</code>, <code className="text-sm">action_diversity</code>) are also effective, showing the strongest discriminative power in Vision mode (0.809-0.816) because the looping behavior upon Vision failure is more pronounced.
            </p>
            <p>
              <strong>Token Level Signals are completely useless</strong> (AUROC ≈ 0.5). The 4B model's probability distribution is too concentrated, making it impossible to extract meaningful uncertainty from logprob/entropy.
            </p>
          </div>
        </section>

        {/* SECTION 5 */}
        <section className="bg-white rounded-2xl p-8 shadow-sm border border-slate-200">
          <SectionHeader num="5" title="Routing Landscape: Tri-Mode Feasible" icon={Layers} />
          
          <div className="space-y-10">
            {/* 5.1 */}
            <div>
              <h3 className="text-2xl font-bold text-slate-800 mb-6">5.1 Theoretical Upper Bound (Oracle Routing)</h3>
              <div className="max-w-3xl overflow-hidden rounded-xl border border-slate-200 shadow-sm">
                <table className="w-full text-left text-base">
                  <thead className="bg-slate-50 text-slate-700 font-semibold border-b border-slate-200">
                    <tr>
                      <th className="p-4"></th>
                      <th className="p-4">B0 (235B)</th>
                      <th className="p-4">B1 (4B)</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100 bg-white">
                    {ORACLE_ROUTING_DATA.map((row, idx) => (
                      <tr key={idx} className={row.isBold ? 'bg-indigo-50/50' : ''}>
                        <td className={`p-4 text-slate-800 ${row.isBold ? 'font-bold' : ''}`}>{row.metric}</td>
                        <td className={`p-4 ${row.isBold ? 'font-bold text-indigo-700' : 'text-slate-600'}`}>{row.b0}</td>
                        <td className={`p-4 ${row.isBold ? 'font-bold text-indigo-700' : 'text-slate-600'}`}>{row.b1}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="mt-4 text-lg text-slate-700">
                B0's routing space is <strong>1.67x</strong> that of B1.
              </p>
            </div>

            {/* 5.2 */}
            <div>
              <h3 className="text-2xl font-bold text-slate-800 mb-6">5.2 Success Set Complementarity (Venn Analysis)</h3>
              
              <div className="grid grid-cols-1 xl:grid-cols-2 gap-8">
                {/* B1 Table */}
                <div>
                  <h4 className="font-bold text-lg text-slate-700 mb-3 bg-slate-100 p-3 rounded-t-xl border-x border-t border-slate-200">
                    B1 Exclusive Success Distribution (adjusted):
                  </h4>
                  <div className="overflow-hidden rounded-b-xl border border-slate-200 shadow-sm">
                    <table className="w-full text-left text-base">
                      <thead className="bg-slate-50 text-slate-600 text-sm">
                        <tr>
                          <th className="p-3">Set</th>
                          <th className="p-3">Count</th>
                          <th className="p-3 w-1/2">Note</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-100 bg-white">
                        {VENN_B1_DATA.map((row, idx) => (
                          <tr key={idx}>
                            <td className={`p-3 text-slate-800 ${row.isBold ? 'font-bold' : ''}`}>{row.set}</td>
                            <td className={`p-3 ${row.isBold ? 'font-bold text-indigo-700' : 'text-slate-600'}`}>{row.count}</td>
                            <td className="p-3 text-sm text-slate-600">{row.note}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>

                {/* B0 Table */}
                <div>
                  <h4 className="font-bold text-lg text-slate-700 mb-3 bg-indigo-50 p-3 rounded-t-xl border-x border-t border-indigo-100 text-indigo-900">
                    B0 Exclusive Success Distribution (adjusted):
                  </h4>
                  <div className="overflow-hidden rounded-b-xl border border-indigo-100 shadow-sm">
                    <table className="w-full text-left text-base">
                      <thead className="bg-indigo-50/50 text-indigo-800 text-sm">
                        <tr>
                          <th className="p-3">Set</th>
                          <th className="p-3">Count</th>
                          <th className="p-3 w-1/2">Note</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-indigo-50 bg-white">
                        {VENN_B0_DATA.map((row, idx) => (
                          <tr key={idx}>
                            <td className={`p-3 text-slate-800 ${row.isBold ? 'font-bold' : ''}`}>{row.set}</td>
                            <td className={`p-3 ${row.isBold ? 'font-bold text-indigo-700' : 'text-slate-600'}`}>{row.count}</td>
                            <td className="p-3 text-sm text-slate-600">{row.note}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            </div>

            {/* 5.3 */}
            <div>
              <h3 className="text-2xl font-bold text-slate-800 mb-6">5.3 Key Findings</h3>
              <ul className="space-y-4 text-lg text-slate-700 list-disc pl-6 marker:text-indigo-500">
                <li>
                  <strong>DOM and Vision successful tasks have absolutely no overlap</strong> (B1 Intersection = 0), making them naturally suited for complementary routing.
                </li>
                <li>
                  <strong>B0 Tri-mode Oracle distribution is more balanced</strong>: SoM 42.6%, Vision 38.2%, DOM 19.1%; Whereas in B1, SoM accounts for 68.2%, making the routing space highly concentrated.
                </li>
                <li>
                  Last week's assumption that DOM was valueless and routing only existed between SoM↔Vision is now corrected to <strong>Tri-Mode Routing</strong>.
                </li>
                <li>
                  <strong>The optimal mode depends on model capability</strong> (Hypothesis from §9 is validated): The stronger the model, the stronger all modes become, and the larger the routing space.
                </li>
              </ul>
            </div>
          </div>
        </section>

        {/* SECTION 6 */}
        <section className="bg-white rounded-2xl p-8 shadow-sm border border-slate-200">
          <SectionHeader num="6" title="Execution Progress & Fixes" icon={Settings} />
          
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
            
            {/* 6.1 Progress */}
            <div>
              <h3 className="text-2xl font-bold text-slate-800 mb-6">6.1 Progress</h3>
              <div className="overflow-hidden rounded-xl border border-slate-200 shadow-sm mb-4">
                <table className="w-full text-left text-base">
                  <thead className="bg-slate-50 text-slate-700 font-semibold border-b border-slate-200">
                    <tr>
                      <th className="p-4">Site</th>
                      <th className="p-4">B0</th>
                      <th className="p-4">B1</th>
                      <th className="p-4">Completed</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100 bg-white">
                    {PROGRESS_DATA.map((row, idx) => (
                      <tr key={idx}>
                        <td className="p-4 font-medium text-slate-800">{row.site}</td>
                        <td className="p-4 text-slate-600">{row.b0}</td>
                        <td className="p-4 text-slate-600">{row.b1}</td>
                        <td className={`p-4 ${row.isBold ? 'font-bold text-indigo-700' : 'text-slate-600'}`}>{row.completed}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="text-lg text-slate-700 px-2 font-medium">VWA 8/18 done, WA 0/18.</p>
            </div>

            {/* 6.2 Fixes */}
            <div>
              <h3 className="text-2xl font-bold text-slate-800 mb-6">6.2 Major Fixes</h3>
              <div className="overflow-hidden rounded-xl border border-slate-200 shadow-sm mb-4">
                <table className="w-full text-left text-base">
                  <thead className="bg-slate-50 text-slate-700 font-semibold border-b border-slate-200">
                    <tr>
                      <th className="p-4 w-1/6">§</th>
                      <th className="p-4 w-1/2">Content</th>
                      <th className="p-4">Data Impact</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100 bg-white">
                    {FIXES_DATA.map((row, idx) => (
                      <tr key={idx}>
                        <td className="p-4 font-mono text-sm text-slate-500">{row.id}</td>
                        <td className="p-4 text-slate-800">{row.desc}</td>
                        <td className="p-4 text-slate-600 text-sm">{row.impact}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="bg-slate-50 p-4 rounded-xl text-slate-700 text-base">
                <strong>New Features:</strong> WebArena integration (§71), scroll cross-model validation (§72), gallery enhancements (§73/§79).
              </div>
            </div>

          </div>
        </section>

        {/* SECTION 7: Next Week's Focus */}
        <section className="bg-indigo-900 text-white rounded-2xl p-8 md:p-12 shadow-xl relative overflow-hidden">
          <div className="absolute top-0 right-0 p-8 opacity-10 pointer-events-none">
            <Calendar className="w-48 h-48" />
          </div>
          
          <div className="relative z-10">
            <h2 className="text-3xl font-bold mb-8 flex items-center gap-3">
              <span className="w-10 h-10 rounded-full bg-indigo-500 text-white flex items-center justify-center font-bold text-xl shadow-sm">7</span>
              Next Week's Focus
            </h2>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-x-12 gap-y-6 text-xl">
              {NEXT_WEEK_FOCUS.map((focus, idx) => (
                <div key={idx} className="flex items-start gap-4">
                  <div className="w-8 h-8 shrink-0 rounded-full bg-indigo-800 border border-indigo-600 flex items-center justify-center text-indigo-300 font-bold text-base mt-0.5">
                    {idx + 1}
                  </div>
                  <div className="leading-relaxed font-medium text-indigo-50">
                    {focus.split('(').map((part, i) => (
                      <span key={i}>
                        {i > 0 && <span className="text-indigo-300 text-lg"> ({part}</span>}
                        {i === 0 && part}
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </section>

      </div>
    </div>
  );
}