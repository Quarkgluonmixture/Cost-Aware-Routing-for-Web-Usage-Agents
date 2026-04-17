import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import {
  AlertTriangle, CheckCircle2, XCircle, Info, Activity, BookOpen, Layers, Zap,
  Bug, MonitorPlay, ListTodo, FileText, Database, GitMerge, FileSearch, ArrowRight, ShieldAlert
} from 'lucide-react';

// ============================================================
// DATA IMPORT — Change this line for each new weekly report
// ============================================================
import data from './data/weekly_4_16.js';

// --- Icon Map (for data-driven icon references) ---
const IconMap = { Zap, ListTodo, Bug, ArrowRight, Activity, BookOpen, Layers, Database, FileText, GitMerge, FileSearch, ShieldAlert, MonitorPlay, AlertTriangle, CheckCircle2 };

// --- Reusable UI Components ---
const SectionHeader = ({ num, title, icon: Icon }) => (
  <div className="flex items-center gap-3 pb-3 border-b-2 border-slate-800 mt-10 mb-6">
    <div className="bg-slate-800 text-white w-9 h-9 rounded flex items-center justify-center font-bold text-lg shadow-sm">
      {num}
    </div>
    <h2 className="text-2xl font-bold text-slate-900 flex items-center gap-2">
      {Icon && <Icon className="w-6 h-6 text-indigo-600" />}
      {title}
    </h2>
  </div>
);

const MetricBox = ({ title, value, subtext, colorClass = "text-indigo-600", bgClass = "bg-white" }) => (
  <div className={`p-4 rounded-xl border border-slate-200 shadow-sm ${bgClass}`}>
    <div className="text-base font-semibold text-slate-500 mb-1">{title}</div>
    <div className={`text-3xl font-bold ${colorClass}`}>{value}</div>
    {subtext && <div className="text-sm text-slate-500 mt-2 font-medium">{subtext}</div>}
  </div>
);

// --- Helper to render HTML strings ---
const Html = ({ children, className = '' }) => (
  <span className={className} dangerouslySetInnerHTML={{ __html: children }} />
);
const HtmlP = ({ children, className = '' }) => (
  <p className={className} dangerouslySetInnerHTML={{ __html: children }} />
);
const HtmlDiv = ({ children, className = '' }) => (
  <div className={className} dangerouslySetInnerHTML={{ __html: children }} />
);

export default function ComprehensiveReport() {
  const { header, disclaimer, srData, significance, costNote, domBugNote,
    falsePositives, oracle, findings, signalEvalData, signalNote,
    routingLitData, ablationData, phantomSom, b0Observations,
    scaffoldFixes, otherInfra, executionStatus, nextMilestones } = data;

  return (
    <div className="min-h-screen bg-[#F8FAFC] p-2 md:p-4 font-sans text-slate-800 leading-relaxed selection:bg-indigo-100">
      <div className="max-w-[1400px] mx-auto space-y-8 bg-white p-4 md:p-8 rounded-2xl shadow-xl border border-slate-100">

        {/* === Header === */}
        <header className="space-y-4">
          <div className="inline-flex items-center gap-2 px-3 py-1 bg-indigo-50 text-indigo-700 text-base font-bold rounded-md mb-2 border border-indigo-100">
            P79 Weekly Report
          </div>
          <h1 className="text-3xl md:text-4xl font-extrabold text-slate-900 tracking-tight leading-snug">
            {header.title}
          </h1>
          <div className="flex flex-wrap gap-x-4 gap-y-2 text-base font-medium text-slate-500 border-b border-slate-200 pb-6">
            <span className="flex items-center gap-1.5"><MonitorPlay className="w-5 h-5"/> {header.date}</span>
            <span className="flex items-center gap-1.5"><Layers className="w-5 h-5"/> {header.phase}</span>
            <span className="flex items-center gap-1.5"><Activity className="w-5 h-5"/> {header.status}</span>
            <span className="flex items-center gap-1.5"><Database className="w-5 h-5"/> {header.models}</span>
          </div>

          {disclaimer && (
            <div className="bg-amber-50/80 border border-amber-200 p-5 rounded-xl flex gap-4 text-base mt-4">
              <ShieldAlert className="w-7 h-7 text-amber-600 shrink-0 mt-0.5" />
              <div className="text-amber-900 space-y-2">
                <p><strong>{disclaimer.title}:</strong> {disclaimer.summary}</p>
                <ul className="list-disc list-inside space-y-1 text-amber-800/90 ml-1">
                  {disclaimer.items.map((item, i) => (
                    <li key={i}><Html>{item}</Html></li>
                  ))}
                </ul>
                <p className="font-medium mt-2">{disclaimer.footer}</p>
              </div>
            </div>
          )}
        </header>

        {/* === Section 1: SR Results === */}
        <section>
          <SectionHeader num="1" title="B1 Classifieds Tri-Mode Full Results" icon={Layers} />
          <p className="mb-4 text-slate-600 text-base font-medium">All three modes completed (234 tasks each). Core metrics:</p>

          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
            <div className="lg:col-span-7 bg-slate-50 p-4 rounded-xl border border-slate-200">
              <div className="h-[320px]">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={srData} margin={{ top: 20, right: 10, left: -10, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
                    <XAxis dataKey="name" axisLine={false} tickLine={false} style={{ fontSize: '14px', fontWeight: 'bold' }} />
                    <YAxis axisLine={false} tickLine={false} tickFormatter={(val) => `${val}%`} style={{ fontSize: '13px' }} />
                    <Tooltip contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)', fontSize: '14px' }} />
                    <Legend wrapperStyle={{ paddingTop: '10px', fontSize: '14px' }} />
                    <Bar dataKey="raw" name="Raw SR" fill="#94a3b8" radius={[4, 4, 0, 0]} maxBarSize={55} />
                    <Bar dataKey="adjusted" name="Adjusted SR" fill="#4f46e5" radius={[4, 4, 0, 0]} maxBarSize={55} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="lg:col-span-5 space-y-4">
              <div className="bg-indigo-50 border border-indigo-100 p-4 rounded-xl text-base">
                <h4 className="font-bold text-indigo-900 mb-2 border-b border-indigo-200 pb-2">Significance Testing (McNemar & Wilcoxon)</h4>
                <ul className="space-y-2 text-indigo-800">
                  {significance.map((s, i) => (
                    <li key={i} className="flex justify-between"><span>{s.pair}:</span> <strong>{s.pValue}</strong></li>
                  ))}
                  <li className="pt-2 mt-2 border-t border-indigo-100 text-indigo-900 leading-relaxed">
                    <Html>{costNote}</Html>
                  </li>
                </ul>
              </div>
              {domBugNote && (
                <div className="text-base text-slate-600 bg-slate-50 p-3 rounded-lg border border-slate-200">
                  <strong>Note:</strong> {domBugNote}
                </div>
              )}
            </div>
          </div>

          <div className="mt-8 space-y-6">
            <p className="text-slate-700 font-medium text-xl border-l-4 border-indigo-500 pl-4 py-1 bg-slate-50">
              The massive drop from Raw to Adjusted SR stems from two classes of False Positives (FP)—this is our most critical methodological finding this week.
            </p>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Visual FP */}
              <div className="border border-red-200 rounded-xl overflow-hidden shadow-sm flex flex-col">
                <div className="bg-red-50 px-4 py-3 font-bold text-red-900 text-lg flex items-center gap-2 border-b border-red-100">
                  <FileSearch className="w-5 h-5 text-red-600" /> {falsePositives.visual.title}
                </div>
                <div className="p-4 text-base text-slate-700 space-y-3 bg-white flex-1">
                  <HtmlP>{falsePositives.visual.description}</HtmlP>
                  <HtmlP>{falsePositives.visual.detail}</HtmlP>
                  <HtmlDiv className="bg-slate-50 p-3 rounded text-sm border border-slate-200 mt-auto">{falsePositives.visual.b0Note}</HtmlDiv>
                </div>
              </div>

              {/* N/A FP */}
              <div className="border border-orange-200 rounded-xl overflow-hidden shadow-sm flex flex-col">
                <div className="bg-orange-50 px-4 py-3 font-bold text-orange-900 text-lg flex items-center gap-2 border-b border-orange-100">
                  <AlertTriangle className="w-5 h-5 text-orange-600" /> {falsePositives.naTask.title}
                </div>
                <div className="p-4 text-base text-slate-700 space-y-3 bg-white flex-1">
                  <HtmlP>{falsePositives.naTask.description}</HtmlP>
                  <ul className="list-disc list-inside text-orange-900/80 pl-1 space-y-1 font-medium">
                    {falsePositives.naTask.types.map((t, i) => (
                      <li key={i}><Html>{t}</Html></li>
                    ))}
                  </ul>
                  <div className="bg-slate-50 p-3 rounded text-sm border border-slate-200 mt-auto">
                    <Html>{falsePositives.naTask.rootCauses}</Html><br/>
                    {falsePositives.naTask.impact}
                  </div>
                </div>
              </div>
            </div>

            {/* Oracle */}
            <div className="bg-gradient-to-r from-indigo-50 to-blue-50 border border-indigo-100 rounded-xl p-5">
              <h3 className="font-bold text-indigo-900 text-xl flex items-center gap-2 mb-4">
                <GitMerge className="w-6 h-6" /> 1.3 Oracle Routing Ceiling
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-4">
                <MetricBox title="Oracle Ceiling (Adjusted)" value={oracle.ceiling} subtext="Best mode per task" bgClass="bg-white/60" />
                <MetricBox title="Headroom (vs SoM)" value={oracle.headroom} colorClass="text-emerald-600" bgClass="bg-white/60" />
                <MetricBox title="Vision-only Successes" value={oracle.visionOnly} subtext="Pure visual tasks where DOM/SoM failed" colorClass="text-purple-600" bgClass="bg-white/60" />
                <MetricBox title="DOM + Vision Overlap" value={oracle.domVisionOverlap} subtext="Completely complementary" colorClass="text-slate-600" bgClass="bg-white/60" />
              </div>
              <HtmlP className="text-base font-semibold text-indigo-900 border-t border-indigo-200 pt-3">
                {'Conclusion: ' + oracle.conclusion}
              </HtmlP>
            </div>
          </div>
        </section>

        {/* === Section 2: Behavioral Findings === */}
        <section>
          <SectionHeader num="2" title="Key Behavioral Findings" icon={Activity} />

          <div className="space-y-6">
            {/* Mirage Effect */}
            <div className="border border-slate-200 rounded-xl p-5 bg-white shadow-sm hover:border-indigo-300 transition-colors">
              <h3 className="text-xl font-bold text-slate-900 mb-3 text-indigo-700">{findings.mirage.title}</h3>
              <HtmlP className="text-base text-slate-700 mb-4">{findings.mirage.intro}</HtmlP>

              <div className="overflow-x-auto rounded-lg border border-slate-200 mb-4">
                <table className="w-full text-base text-left">
                  <thead className="bg-slate-50 text-slate-700 border-b border-slate-200">
                    <tr><th className="px-4 py-2.5">Mirage Paper</th><th className="px-4 py-2.5">B1 Observation</th></tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100 text-slate-600">
                    {findings.mirage.comparisons.map((c, i) => (
                      <tr key={i}>
                        <td className="px-4 py-2.5">{c.paper}</td>
                        <td className="px-4 py-2.5 font-medium">{c.observation}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="text-base text-slate-700 bg-indigo-50 p-3 rounded border border-indigo-100">
                <strong>Explanatory Power:</strong> {findings.mirage.explanatoryPower}
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Cognitive Gap */}
              <div className="border border-slate-200 rounded-xl p-5 bg-white shadow-sm hover:border-emerald-300 transition-colors">
                <h3 className="text-xl font-bold text-slate-900 mb-3 text-emerald-700">{findings.cognitiveGap.title}</h3>
                <div className="space-y-3 text-base text-slate-700">
                  <p><strong>{findings.cognitiveGap.caseStudy}</strong></p>
                  <ul className="list-disc pl-5 space-y-1">
                    <li><strong>DOM/SoM:</strong> <Html>{findings.cognitiveGap.domSom}</Html></li>
                    <li><strong>Vision:</strong> {findings.cognitiveGap.vision}</li>
                  </ul>
                  <p className="pt-2 border-t border-slate-100">{findings.cognitiveGap.broader}</p>
                </div>
              </div>

              {/* Self-Correction */}
              <div className="border border-slate-200 rounded-xl p-5 bg-white shadow-sm hover:border-rose-300 transition-colors">
                <h3 className="text-xl font-bold text-slate-900 mb-3 text-rose-700">{findings.selfCorrection.title}</h3>
                <div className="space-y-3 text-base text-slate-700">
                  <HtmlP>{findings.selfCorrection.description}</HtmlP>
                  <HtmlP className="bg-rose-50 p-3 rounded border border-rose-100">{findings.selfCorrection.keyPoint}</HtmlP>
                  <p>{findings.selfCorrection.implication}</p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* === Section 3: Routing === */}
        <section>
          <SectionHeader num="3" title="Routing Theoretical Framework" icon={BookOpen} />

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
            <div className="space-y-4">
              <h3 className="font-bold text-lg text-slate-800 flex items-center gap-2"><Zap className="w-5 h-5 text-indigo-500"/> 3.1 Signal Evaluation (Preliminary)</h3>
              <div className="border border-slate-200 rounded-lg overflow-hidden">
                <table className="w-full text-sm text-left">
                  <thead className="bg-slate-100 text-slate-700">
                    <tr><th className="px-3 py-2.5">Signal Type</th><th className="px-3 py-2.5">Specific Signal</th><th className="px-3 py-2.5">AUROC</th><th className="px-3 py-2.5">Conclusion</th></tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100 bg-white">
                    {signalEvalData.map((row, i) => (
                      <tr key={i}>
                        <td className="px-3 py-2.5 font-medium">{row.type}</td>
                        <td className="px-3 py-2.5">{row.signal}</td>
                        <td className="px-3 py-2.5 font-bold text-indigo-600 whitespace-nowrap">{row.auroc}</td>
                        <td className="px-3 py-2.5 text-slate-600">{row.conclusion}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <HtmlP className="text-sm text-slate-600 leading-relaxed">{signalNote}</HtmlP>
            </div>

            <div className="space-y-4">
              <h3 className="font-bold text-lg text-slate-800 flex items-center gap-2"><FileText className="w-5 h-5 text-indigo-500"/> 3.2 Routing Lit Review (\u00a724)</h3>
              <p className="text-sm text-slate-500">Systematic review of ~1400 papers (2023-2026), establishing priority:</p>
              <div className="border border-slate-200 rounded-lg overflow-hidden">
                <table className="w-full text-sm text-left">
                  <thead className="bg-slate-100 text-slate-700">
                    <tr><th className="px-3 py-2.5">Signal</th><th className="px-3 py-2.5">Status</th><th className="px-3 py-2.5">Literature Support</th></tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100 bg-white">
                    {routingLitData.map((row, i) => (
                      <tr key={i}>
                        <td className="px-3 py-2.5 font-medium">{row.signal}</td>
                        <td className={`px-3 py-2.5 font-bold ${row.status.includes('Impl')||row.status.includes('Analysis') ? 'text-emerald-600' : 'text-slate-400'}`}>{row.status}</td>
                        <td className="px-3 py-2.5">{row.reference}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Phantom-SoM */}
          <div className="border-2 border-indigo-100 bg-white rounded-xl overflow-hidden shadow-sm">
            <div className="bg-indigo-50/50 px-5 py-4 border-b border-indigo-100">
              <h3 className="text-xl font-bold text-indigo-900">3.3 Phantom-SoM Ablation Design (\u00a725)</h3>
              <HtmlP className="text-base text-indigo-800/80 mt-1">{phantomSom.intro}</HtmlP>
            </div>

            <div className="p-5">
              <p className="text-base text-slate-700 mb-4">We designed 5 ablation groups to isolate three confounding variables (prompt/scaffold, text format, image implication):</p>
              <div className="overflow-x-auto rounded-lg border border-slate-200 mb-4">
                <table className="w-full text-base text-left whitespace-nowrap">
                  <thead className="bg-slate-800 text-white">
                    <tr><th className="px-4 py-2.5">Group</th><th className="px-4 py-2.5">Prompt</th><th className="px-4 py-2.5">Text Content</th><th className="px-4 py-2.5">Visuals</th><th className="px-4 py-2.5">Isolated Variable</th></tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {ablationData.map((row, i) => (
                      <tr key={i} className={row.bg}>
                        <td className="px-4 py-2.5 font-bold">{row.group}</td>
                        <td className="px-4 py-2.5">{row.prompt}</td>
                        <td className="px-4 py-2.5">{row.text}</td>
                        <td className="px-4 py-2.5">{row.image}</td>
                        <td className="px-4 py-2.5">{row.variable}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="bg-slate-50 p-4 rounded-lg text-base text-slate-700 border border-slate-200 space-y-2">
                <HtmlP>{phantomSom.causalLogic}</HtmlP>
                <HtmlP className="text-indigo-800 font-medium">{phantomSom.critical}</HtmlP>
                <HtmlP className="text-sm text-slate-500 mt-2 pt-2 border-t border-slate-200">{phantomSom.novelty}</HtmlP>
              </div>
            </div>
          </div>
        </section>

        {/* === Section 4: B0 === */}
        <section>
          <SectionHeader num="4" title="B0 Preliminary Observations (Qwen3-235B)" icon={Database} />
          <p className="text-base text-slate-600 mb-4 font-medium">Initial look at B0 (Qwen3-235B-A22B) Classifieds DOM (affected by parse_error, pending rerun):</p>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {b0Observations.map((obs, i) => {
              const ObsIcon = IconMap[obs.icon] || Zap;
              const colorMap = { indigo: 'text-indigo-500', orange: 'text-orange-500', red: 'text-red-500', slate: 'text-slate-500' };
              return (
                <div key={i} className="bg-white border border-slate-200 p-4 rounded-xl flex gap-4 hover:shadow-md transition-shadow">
                  <ObsIcon className={`w-7 h-7 ${colorMap[obs.color] || 'text-slate-500'} shrink-0`} />
                  <div>
                    <h4 className="font-bold text-lg text-slate-800 mb-1">{obs.title}</h4>
                    <HtmlP className="text-base text-slate-600">{obs.description}</HtmlP>
                  </div>
                </div>
              );
            })}
          </div>
        </section>

        {/* === Section 5 & 6 === */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mt-10">
          <section>
            <SectionHeader num="5" title="Scaffold Fixes Summary" icon={Bug} />
            <p className="text-base text-slate-600 mb-3">Numerous scaffold bugs fixed this week, most necessitating reruns:</p>
            <div className="border border-slate-200 rounded-lg overflow-hidden">
              <table className="w-full text-sm text-left">
                <thead className="bg-slate-100 text-slate-700">
                  <tr><th className="px-3 py-2.5 w-12">\u00a7</th><th className="px-3 py-2.5">Issue</th><th className="px-3 py-2.5">Status</th></tr>
                </thead>
                <tbody className="divide-y divide-slate-100 bg-white">
                  {scaffoldFixes.map((fix, i) => (
                    <tr key={i}>
                      <td className="px-3 py-2.5 font-mono text-slate-400">{fix.id}</td>
                      <td className="px-3 py-2.5">
                        <div className="font-medium text-slate-800">{fix.issue}</div>
                        <div className="text-slate-500 mt-0.5">Impact: {fix.impact}</div>
                      </td>
                      <td className={`px-3 py-2.5 font-bold whitespace-nowrap ${fix.status.includes('rerun')||fix.status.includes('pending') ? 'text-amber-600' : 'text-emerald-600'}`}>
                        {fix.status}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="mt-4 p-3 bg-slate-50 border border-slate-200 rounded-lg text-sm text-slate-600 leading-relaxed">
              <strong>Other Infrastructure Complete:</strong> {otherInfra}
            </div>
          </section>

          <section>
            <SectionHeader num="6" title="Status & Next Steps" icon={ListTodo} />

            <div className="space-y-3 mb-8">
              <h3 className="text-base font-bold text-slate-500 uppercase tracking-wider mb-2">Execution Queue</h3>
              {executionStatus.map((item, i) => (
                <div key={i} className={`p-3 rounded-lg border flex justify-between items-center ${item.color}`}>
                  <div>
                    <div className="font-bold text-base">{item.task}</div>
                    <div className="text-sm opacity-80 mt-0.5">{item.note}</div>
                  </div>
                  <div className="text-base font-bold px-2 py-1 bg-white/50 rounded">{item.status}</div>
                </div>
              ))}
            </div>

            <div className="bg-indigo-900 text-white p-5 rounded-xl shadow-lg">
              <h3 className="font-bold text-lg text-indigo-100 mb-4 flex items-center gap-2">
                <CheckCircle2 className="w-6 h-6 text-indigo-400" /> Next Week's Milestones
              </h3>
              <ul className="space-y-3 text-base font-medium">
                {nextMilestones.map((milestone, i) => (
                  <li key={i} className="flex gap-3 items-start">
                    <span className="bg-indigo-800 w-6 h-6 rounded-full flex items-center justify-center shrink-0 text-sm text-indigo-300">{i + 1}</span>
                    {milestone}
                  </li>
                ))}
              </ul>
            </div>
          </section>
        </div>

      </div>
    </div>
  );
}
