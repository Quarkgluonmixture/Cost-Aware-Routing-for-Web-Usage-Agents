# Sustainability Measurement Methodology for AI-Agent Inference

## Purpose

This note establishes a defensible methodology for discussing sustainability in an AI-agent dissertation that measures:

- token usage;
- monetary inference cost;
- latency;
- GPU-side operational energy.

The central methodological constraint is:

> **Computational cost proxies must not be silently re-labelled as environmental impact.**

In particular:

> **A 30% reduction in token count does not imply a 30% reduction in energy use, and neither implies a 30% reduction in CO₂e.**

A defensible measurement hierarchy is:

\[
\text{tokens / FLOPs / latency / \$}
\rightarrow
\text{computational and economic efficiency}
\]

\[
\text{measured electrical energy}
\rightarrow
\text{operational energy demand}
\]

\[
\text{energy} \times \text{electricity carbon intensity}
\rightarrow
\text{operational GHG emissions estimate}
\]

\[
\text{operational emissions + facility overhead + embodied emissions + wider lifecycle effects}
\rightarrow
\text{broader carbon / environmental footprint}
\]

The dissertation should keep these layers conceptually and terminologically separate.

---

# 1. Bottom-line recommendations

For the current measurement setup, the safest terminology is:

- **token count** → computational demand / computational efficiency;
- **latency** → runtime efficiency;
- **monetary cost** → economic efficiency;
- **GPU telemetry-derived energy** → GPU-device operational energy;
- **GPU energy × explicit grid-carbon-intensity factor** → GPU-device operational emissions estimate;
- **do not call the above a total carbon footprint, SCI score, or comprehensive environmental impact assessment** unless the missing system-boundary components are added.

The most useful formal methodological anchor is:

> **ISO/IEC 21031:2024 — Software Carbon Intensity (SCI) specification**

Its basic decomposition is:

\[
O = E \times I
\]

where:

- \(O\) = operational emissions;
- \(E\) = energy consumed;
- \(I\) = electricity carbon intensity.

For full SCI accounting:

\[
SCI = \frac{O + M}{R}
\]

where:

- \(M\) = embodied emissions associated with hardware/infrastructure;
- \(R\) = the chosen functional unit.

This decomposition is important because **token count is not itself a carbon-emissions term**. It may be used as a workload or functional-unit descriptor, but CO₂e requires an energy term and a carbon-intensity term.

---

# 2. Evidence table

| 问题 | 结论 | 出处（标题 + arXiv ID/DOI/URL） | 出处类型 | 我该怎么引用它 |
|---|---|---|---|---|
| **1. token / FLOPs 能否直接推能耗？** | **不存在一个通用、硬件无关的 token→kWh 或 FLOPs→kWh 换算。** 对固定模型、硬件、precision、batching、serving stack 和 workload，可以经验测得 energy/token；但这个比率不应跨系统直接外推。ACL 2025 的实证结果显示，基于 FLOPs 或理论 GPU utilisation 的朴素推理能耗估计可能明显偏离实际能耗，而且优化收益依赖 workload、software stack 和 accelerator。 | Fernandez et al., **Energy Considerations of Large Language Model Inference and Efficiency Optimizations**, DOI: `10.18653/v1/2025.acl-long.1563`; ACL Anthology: https://aclanthology.org/2025.acl-long.1563/ | 论文 | 用来支持：“**token count is a computational workload indicator, not a direct measurement of energy consumption**.” 这是反驳“token↓30%=energy↓30%”最直接的来源之一。 |
| **1. GPU-hour 能否直接视为能耗？** | GPU-hour 是 activity / compute quantity，不是 energy。经典 Green Algorithms 方法仍需 runtime、GPU/CPU 类型和数量、功率、utilisation、memory，以及 PUE 等输入。 | Lannelongue et al., **Green Algorithms: Quantifying the Carbon Footprint of Computation**, arXiv: `2007.07610`, DOI: `10.1002/advs.202100707`; https://arxiv.org/abs/2007.07610 | 方法论文 | 如果已经有直接 GPU telemetry，优先积分实际功率而不是退回 GPU-hour×TDP。该文适合说明“GPU-hours 本身不等于能耗，以及估算需要哪些假设”。 |
| **1. 经典 ML 能耗/碳核算 precedent** | Strubell et al. 通过测量 CPU/GPU 平均功率、运行时间，并结合 PUE 与电网排放因子估算训练能耗与碳排。这是 ML 领域早期重要 precedent，但其中某些参数使用平均值，因此今天若有更细粒度数据，不应机械照抄。 | Strubell et al., **Energy and Policy Considerations for Deep Learning in NLP**, arXiv: `1906.02243`, DOI: `10.18653/v1/P19-1355`; https://aclanthology.org/P19-1355/ | 论文 | 可用于 Related Work / historical precedent；方法部分更建议以 SCI + Green Algorithms 为主规范。 |
| **1. 能耗如何推 operational CO₂e？** | 主流结构是 **energy × electricity carbon intensity**。SCI 明确定义 \(O=E\times I\)。因此从 token/GPU-hour 到 CO₂e 中间不能跳过 energy 与 grid carbon intensity。 | **ISO/IEC 21031:2024 — Software Carbon Intensity specification**; ISO: https://www.iso.org/standard/86612.html ; public SCI spec: https://sci.greensoftware.foundation/ | 国际标准 | 作为 sustainability methodology 的主规范。引用它来明确区分 compute proxy、measured energy 和 operational emissions。 |
| **2. 英国 GB 电网碳强度高时间分辨率来源** | NESO Carbon Intensity API 为 Great Britain 提供全国及区域电网碳强度数据。区域模型覆盖 14 个 GB region，并以约 30-minute 粒度提供 forecast / estimated intensity，可按 region/postcode 查询。 | NESO, **Carbon Intensity API**: https://api.carbonintensity.org.uk/ ; methodology/site: https://carbonintensity.org.uk/ | 官方机构/API | 若 GPU 实验在 GB，保存 experiment timestamp、region、使用的 API interval/value、实际或预测值，并明确其定义边界。 |
| **2. 英国官方年度 GHG conversion factors** | UK Government / DESNZ 提供年度 GHG conversion factors。适合没有精确实验时刻、只能进行年度 location-based activity accounting 的情况，但时间粒度明显粗于实时/半小时数据。 | DESNZ, **Greenhouse gas reporting: conversion factors 2026**: https://www.gov.uk/government/publications/greenhouse-gas-reporting-conversion-factors-2026 | 政府方法文件 | 如果用年度因子，应明确写 “2026 UK conversion factor”，不要表述成“实验发生时的实时 grid carbon intensity”。 |
| **2. 全球 / 云 region 电网碳强度来源** | Electricity Maps 提供按 geographic zone / data-centre region 查询的电力碳强度 API，并区分 direct 与 lifecycle emission factors；时间粒度可达到小时甚至更细。 | **Electricity Maps API**: https://app.electricitymaps.com/developer-hub/api/reference | 工具/API 文档 | 报告 zone、timestamp、temporal resolution、emission-factor type，以及是否使用 flow-traced / estimated 数据。不要把 direct 与 lifecycle 因子混用。 |
| **2. 全球月度/年度公开电力数据** | Ember API 提供国家/地区层面的 monthly/yearly carbon-intensity related data，适合无法恢复精确运行时间、只能确定国家/月份的情况。 | **Ember API**: https://api.ember-energy.org/v1/docs | 开放数据/API | 引用时注明 entity、dataset、temporal resolution、访问日期和许可。不要将月均值描述为实验时刻的 carbon intensity。 |
| **2. 美国电网数据** | EPA eGRID 提供 plant/state/balancing-area/eGRID-subregion 等层级的电网排放因子和 emission rates，是美国 location-based grid accounting 的标准公开来源之一。 | US EPA, **eGRID**: https://www.epa.gov/egrid/summary-data | 政府数据库 | 报告 dataset year/revision 和使用的 eGRID subregion。 |
| **2. average vs marginal carbon intensity** | 对 footprint / inventory，average location-based grid factor 很常见；若声称“某优化造成/避免了 X kg 排放”，则进入 consequential / avoided-emissions accounting，方法要求更高。SCI 允许不同 grid-intensity methodologies，但必须明确说明选择。 | **GHG Protocol Scope 2 Guidance**: https://ghgprotocol.org/scope-2-guidance ; ISO/IEC 21031:2024 | GHG accounting guidance + 国际标准 | 对当前论文，优先使用 “location-based operational emissions estimate”，避免未经额外方法支持使用 “avoided emissions”。 |
| **3. 当前 software-carbon 规范** | **存在正式国际标准：ISO/IEC 21031:2024 SCI。** 要求声明 measurement boundary、functional unit、quantification methodology，并区分 operational 与 embodied emissions。 | **ISO/IEC 21031:2024 — Software Carbon Intensity specification**: https://www.iso.org/standard/86612.html | 国际标准 | Sustainability Methodology 最值得作为规范主引用。 |
| **3. PUE 规范** | PUE 是 data-centre facility energy 与 IT-equipment energy 关系的重要 KPI。仅测 GPU device power 时，并未自动包含 cooling、power distribution 等 facility overhead。 | **ISO/IEC 30134-2:2026 — Power usage effectiveness (PUE)**: https://www.iso.org/standard/30134-2 | 国际标准 | 用来解释为什么 GPU telemetry ≠ total data-centre energy。缺 PUE 时，建议声明边界而不是私自填一个行业平均值。 |
| **3. ML-specific reporting practice** | ML 社区存在高引用 reporting frameworks，但它们不是取代 ISO/SCI 的统一强制国际标准。Henderson et al. 推动系统报告 energy/carbon；GREENER 提出透明度、估算和报告原则。 | Henderson et al., **Towards the Systematic Reporting of the Energy and Carbon Footprints of Machine Learning**, arXiv: `2002.05651`; Lannelongue et al., **GREENER principles for environmentally sustainable computational science**, DOI: `10.1038/s43588-023-00461-y` | 论文 / best-practice framework | 可称为 ML / computational-science reporting practice，不要称为 formal international standard。 |
| **3/5. broader environmental impact / LCA 边界** | 如果论文使用 “environmental impact” 或 “life-cycle footprint” 这种更强表述，规范边界明显大于 GPU electricity，通常需要 lifecycle system boundary、inventory、impact assessment、hardware/infrastructure 等。 | **ITU-T L.1410 (11/2024), Methodology for environmental life cycle assessments of ICT goods, networks and services**: https://www.itu.int/rec/T-REC-L.1410-202411-I/en | 国际 Recommendation / LCA methodology | 非常适合用于声明：**本研究不是完整 LCA。** |
| **4. computational efficiency 与 environmental impact 必须区分** | 计算效率指标不是环境影响指标。真实能耗并非只由 FLOPs/token 决定；CO₂e 还取决于 energy source、地点、时间和 facility；full footprint 进一步包括 embodied emissions。 | Fernandez et al. 2025, DOI `10.18653/v1/2025.acl-long.1563`; Patterson et al., **Carbon Emissions and Large Neural Network Training**, arXiv: `2104.10350`; ISO/IEC 21031:2024 | 论文 + 标准 | 推荐核心句：“**We treat token count as a measure of computational demand, not as a direct proxy for energy use or GHG emissions.**” |
| **4. 生命周期边界是否真的影响结果？** | 会。BLOOM carbon-footprint work 表明，仅计 dynamic compute 与纳入更广的设备/运营生命周期后，排放估计可以显著变化。Chasing Carbon 也明确强调 operational 与 embodied carbon 的区分。 | Luccioni et al., **Estimating the Carbon Footprint of BLOOM**, arXiv: `2211.02001`; Gupta et al., **Chasing Carbon: The Elusive Environmental Footprint of Computing**, arXiv: `2011.02839`, DOI: `10.1109/MM.2022.3163226` | 论文 | 用于 Limitations，说明为何不应把 GPU-side energy 叫 total carbon footprint。 |
| **5. 只有 GPU energy、没有 PUE/制造/网络时怎么办？** | **显式缩窄 system boundary，而不是补假设后装作 full footprint。** 最稳妥名称是 GPU-device operational energy；若乘显式 grid CI，可称 GPU-device operational emissions estimate。 | ISO/IEC 21031:2024; Green Algorithms DOI `10.1002/advs.202100707`; ISO/IEC 30134-2:2026 | 标准 + 方法论文 | 避免使用 “total system energy”, “SCI score”, “carbon footprint of the agent”, “comprehensive environmental impact” 等超出边界的称呼。 |

---

# 3. Methodological hierarchy

## 3.1 Computational efficiency

Suitable measurements:

- input tokens;
- output tokens;
- total tokens;
- number of model calls;
- latency;
- GPU-hours as a compute-activity measure;
- monetary inference cost;
- FLOPs, if available.

These quantities describe **computational or economic demand**.

They do **not**, by themselves, measure:

- electrical energy;
- data-centre energy;
- operational CO₂e;
- embodied carbon;
- total environmental impact.

A computational optimisation may reduce energy use, but the magnitude must be measured or modelled under a specified hardware/software configuration.

---

## 3.2 Device-level operational energy

If GPU telemetry provides instantaneous or interval power \(P(t)\), energy is conceptually:

\[
E_{\mathrm{GPU}} = \int P_{\mathrm{GPU}}(t)\,dt
\]

or, for sampled telemetry,

\[
E_{\mathrm{GPU}}
\approx
\sum_i P_i \Delta t_i.
\]

Convert to kWh before combining with grid carbon intensity.

This quantity is best called:

> **GPU-device operational energy**

unless the monitoring setup demonstrably includes other components.

It should not automatically be called:

- machine energy;
- server energy;
- data-centre energy;
- total inference energy.

Those broader terms require corresponding measurement coverage.

---

## 3.3 Facility energy

Data-centre overhead may include:

- cooling;
- UPS and conversion losses;
- power distribution;
- lighting;
- other facility infrastructure.

PUE is commonly defined as the relation between total facility energy and IT-equipment energy.

A simplified modelling form is:

\[
E_{\mathrm{facility}}
\approx
E_{\mathrm{IT}} \times PUE.
\]

However, if the dissertation does **not** know the deployment facility's PUE, the most defensible procedure is generally:

1. do not invent a facility-specific value;
2. report device-level energy separately;
3. state that facility overhead is outside the measurement boundary.

---

# 4. Converting energy to operational GHG emissions

With an explicitly selected grid-emissions factor:

\[
O = E \times I
\]

where:

- \(E\): electrical energy, usually kWh;
- \(I\): grid carbon intensity, typically gCO₂e/kWh or kgCO₂e/kWh;
- \(O\): operational GHG emissions attributable under the stated accounting method.

Example:

\[
O_{\mathrm{GPU}}
=
E_{\mathrm{GPU}}
\times
I_{\mathrm{grid}}.
\]

This should be described as something like:

> **location-based GPU-device operational GHG emissions estimate**

rather than:

> carbon footprint of the AI agent.

The latter implies a broader lifecycle/system boundary.

---

# 5. Choosing grid carbon-intensity data

## 5.1 Preferred principle

Use the **highest defensible temporal and geographical resolution available for the actual experiment**.

Ideally record:

- experiment start/end timestamp;
- physical region or cloud region;
- electricity grid zone;
- grid-carbon-intensity source;
- temporal resolution;
- direct vs lifecycle emission factor;
- actual/estimated/forecast status;
- retrieval date/API version if relevant.

---

## 5.2 Great Britain

### NESO Carbon Intensity API

Source:

- https://api.carbonintensity.org.uk/
- https://carbonintensity.org.uk/

Good choice when:

- the experiment physically ran in Great Britain;
- timestamps are available;
- regional or national half-hourly resolution is useful.

Recommended reporting pattern:

> Electricity carbon intensity was matched to the experiment timestamp using the NESO Carbon Intensity API at [national/region] resolution and [30-minute] intervals.

Do not silently convert this into a claim about complete lifecycle CO₂e if the selected NESO metric has a narrower generation-emissions definition.

---

## 5.3 UK Government annual conversion factors

Source:

- https://www.gov.uk/government/publications/greenhouse-gas-reporting-conversion-factors-2026

Good choice when:

- experiment timestamps cannot be reconstructed;
- only UK-wide annual accounting is defensible.

Trade-off:

- authoritative;
- but coarse temporal resolution.

If used, explicitly say it is an **annual UK conversion factor**, not instantaneous grid intensity.

---

## 5.4 Electricity Maps

Source:

- https://app.electricitymaps.com/developer-hub/api/reference

Useful for:

- cloud regions;
- international experiments;
- timestamp-specific estimates.

Important fields to report:

- electricity zone;
- temporal resolution;
- direct vs lifecycle emission factor;
- flow-tracing choice;
- estimated vs measured status.

---

## 5.5 Ember

Source:

- https://api.ember-energy.org/v1/docs

Useful for:

- country/month/year level comparisons;
- cases where exact runtime location/time cannot be recovered.

Not ideal for:

- attributing emissions to a short inference run if hourly/half-hourly data exist.

---

## 5.6 US EPA eGRID

Source:

- https://www.epa.gov/egrid/summary-data

Useful for:

- US location-based accounting;
- region/subregion emission rates.

Report:

- eGRID dataset year;
- revision/version;
- subregion.

---

# 6. Average vs marginal carbon intensity

This distinction matters.

## Average / location-based intensity

Answers approximately:

> “Given electricity consumption at this place and accounting period, what emissions intensity is associated with the grid mix?”

Appropriate for:

- footprint reporting;
- attributional accounting;
- descriptive operational-emissions estimates.

---

## Marginal intensity

Answers approximately:

> “What generating resources would likely respond to a marginal change in electricity demand?”

More relevant when claiming:

- emissions avoided by an optimisation;
- causal emissions reduction;
- scheduling-induced emissions change.

Therefore the dissertation should preferably avoid a strong sentence such as:

> “The router avoided X kg CO₂e.”

unless a consequential methodology using an appropriate marginal factor has been justified.

Safer:

> “Under a location-based accounting method, the measured GPU-energy reduction corresponds to an estimated reduction of X gCO₂e under the specified grid-intensity assumptions.”

---

# 7. Current standards / methodological references

## 7.1 ISO/IEC 21031:2024

**Title:** Software Carbon Intensity (SCI) specification  
**Type:** International standard  
**URL:** https://www.iso.org/standard/86612.html  
**Public specification:** https://sci.greensoftware.foundation/

Most important points for this dissertation:

- define the software boundary;
- define a functional unit;
- quantify energy;
- specify electricity carbon intensity;
- distinguish operational and embodied emissions;
- disclose methodology and assumptions.

Use this as the main normative anchor.

---

## 7.2 ISO/IEC 30134-2:2026

**Title:** Information technology — Data centres key performance indicators — Part 2: Power usage effectiveness (PUE)  
**Type:** International standard  
**URL:** https://www.iso.org/standard/30134-2

Use it to justify the statement:

> GPU-device energy does not automatically include data-centre facility overhead.

---

## 7.3 ITU-T L.1410 (11/2024)

**Title:** Methodology for environmental life cycle assessments of information and communication technology goods, networks and services  
**Type:** ITU Recommendation / lifecycle methodology  
**URL:** https://www.itu.int/rec/T-REC-L.1410-202411-I/en

Use it when distinguishing:

- operational energy/emissions;
- lifecycle carbon;
- broader environmental impact.

This is particularly useful for saying:

> This dissertation does not perform a full lifecycle assessment.

---

## 7.4 GHG Protocol Scope 2 Guidance

**URL:** https://ghgprotocol.org/scope-2-guidance

Use for:

- location-based electricity accounting;
- discussion of electricity-emissions factors.

Do not use it to overstate causal avoided emissions from reduced computation.

---

# 8. Key methodological papers

## 8.1 Fernandez et al. (ACL 2025)

**Title:** Energy Considerations of Large Language Model Inference and Efficiency Optimizations  
**DOI:** `10.18653/v1/2025.acl-long.1563`  
**URL:** https://aclanthology.org/2025.acl-long.1563/

Why it matters:

- directly studies LLM inference energy;
- shows theoretical proxies such as FLOPs / theoretical utilisation are insufficient for reliable energy estimation;
- demonstrates dependence on workload geometry, software stack, hardware and optimisation choices.

Best use:

> Support the distinction between computational workload and measured energy.

---

## 8.2 Lannelongue et al. — Green Algorithms

**Title:** Green Algorithms: Quantifying the Carbon Footprint of Computation  
**arXiv:** `2007.07610`  
**DOI:** `10.1002/advs.202100707`  
**URL:** https://arxiv.org/abs/2007.07610

Why it matters:

- supplies a transparent computational-energy/carbon estimation methodology;
- requires runtime, hardware power and utilisation assumptions;
- introduces facility/PUE considerations.

Best use:

> Explain why GPU-hours alone do not equal kWh or carbon emissions.

---

## 8.3 Strubell et al. (ACL 2019)

**Title:** Energy and Policy Considerations for Deep Learning in NLP  
**arXiv:** `1906.02243`  
**DOI:** `10.18653/v1/P19-1355`  
**URL:** https://aclanthology.org/P19-1355/

Why it matters:

- influential ML precedent for reporting energy/carbon;
- combines measured compute power, runtime, PUE and electricity emissions factors.

Best use:

> Historical ML methodology / Related Work.

---

## 8.4 Henderson et al. (JMLR)

**Title:** Towards the Systematic Reporting of the Energy and Carbon Footprints of Machine Learning  
**arXiv:** `2002.05651`  
**URL:** https://arxiv.org/abs/2002.05651

Why it matters:

- advocates systematic reporting of ML energy/carbon;
- useful as a reporting best-practice reference.

Best use:

> Reporting methodology rather than formal international standard.

---

## 8.5 GREENER principles

**Title:** GREENER principles for environmentally sustainable computational science  
**DOI:** `10.1038/s43588-023-00461-y`

Why it matters:

- emphasises transparent measurement, estimation and disclosure.

Best use:

> Reporting/checklist-style support for sustainability methodology.

---

## 8.6 Patterson et al.

**Title:** Carbon Emissions and Large Neural Network Training  
**arXiv:** `2104.10350`  
**URL:** https://arxiv.org/abs/2104.10350

Why it matters:

- shows that operational emissions strongly depend on hardware efficiency, data-centre efficiency and geographic/electricity conditions.

Best use:

> Reinforce the point that equivalent computation need not imply equivalent carbon emissions.

---

## 8.7 Luccioni et al. — BLOOM

**Title:** Estimating the Carbon Footprint of BLOOM, a 176B Parameter Language Model  
**arXiv:** `2211.02001`  
**URL:** https://arxiv.org/abs/2211.02001

Why it matters:

- explicitly expands carbon accounting beyond dynamic accelerator electricity;
- illustrates how widening system/lifecycle boundaries changes total estimates.

Best use:

> Limitations and lifecycle-boundary discussion.

---

## 8.8 Gupta et al. — Chasing Carbon

**Title:** Chasing Carbon: The Elusive Environmental Footprint of Computing  
**arXiv:** `2011.02839`  
**DOI:** `10.1109/MM.2022.3163226`

Why it matters:

- emphasises operational versus embodied carbon;
- shows why computing-related environmental impact cannot be reduced to runtime compute metrics alone.

Best use:

> Broader sustainability framing and limitations.

---

# 9. Recommended terminology for the dissertation

## Preferred

- computational efficiency;
- inference efficiency;
- token demand;
- monetary inference cost;
- latency;
- measured GPU-device operational energy;
- location-based operational GHG emissions estimate;
- device-level measurement boundary;
- operational-energy reduction under the evaluated configuration;
- potential operational-emissions implications;
- carbon-intensity-dependent emissions estimate.

---

## Avoid unless additional measurements support them

- total energy consumption;
- total system energy;
- carbon footprint of the agent;
- environmental impact of the agent;
- sustainable because it uses fewer tokens;
- 30% fewer tokens therefore 30% lower carbon emissions;
- avoided emissions;
- SCI score;
- lifecycle emissions.

---

# 10. Recommended dissertation logic

A strong argument is:

1. **Tokens, latency and dollar cost quantify inference efficiency.**
2. **GPU telemetry independently measures accelerator operational energy.**
3. This allows checking whether computational savings translate into actual device-level energy savings under the evaluated configuration.
4. **No proportional equivalence is assumed.**
5. If location/time-specific grid carbon intensity is available, GPU energy may be converted into a **bounded operational-emissions estimate**.
6. PUE, CPU/memory, networking and embodied hardware emissions remain outside the measurement boundary.
7. Therefore, the study contributes evidence about the **operational sustainability implications** of the router without claiming a complete lifecycle environmental assessment.

This gives the sustainability part of the MSc dissertation substantive methodological content without overstating what was measured.

---

# 11. Draft A — measurement boundary

> **Draft for adaptation into the dissertation**

We report **GPU-device operational energy** measured during inference, rather than a full-system or life-cycle carbon footprint. The measurement boundary is restricted to the accelerator energy captured by GPU telemetry and therefore excludes host CPU and memory consumption, storage and networking, and data-centre overheads such as cooling and power distribution, for which Power Usage Effectiveness (PUE) was not available. It also excludes embodied emissions associated with hardware manufacture, infrastructure, and end-of-life. Consequently, the reported energy values should be interpreted as a directly measured, device-level operational quantity, not as total facility energy, a Software Carbon Intensity (SCI) score, or a comprehensive assessment of environmental impact (ISO/IEC 21031:2024; Lannelongue et al., 2021; ITU-T L.1410, 2024).

---

# 12. Draft B — linking inference cost to environmental impact without overclaiming

> **Draft for adaptation into the dissertation**

Reductions in token usage, monetary cost, and latency are interpreted here as improvements in **computational and economic efficiency**, rather than as direct reductions in greenhouse-gas emissions. Token count does not uniquely determine energy consumption: inference energy depends additionally on workload characteristics, the software stack, serving configuration, and hardware accelerator, and operational emissions further depend on the carbon intensity of the electricity consumed. We therefore do not infer a proportional reduction in CO₂e from a proportional reduction in token usage. Where reductions are also observed in directly measured GPU energy, they provide evidence of lower **device-level operational energy demand** under the evaluated conditions; any corresponding emissions reduction requires an explicitly specified electricity carbon-intensity factor and remains subject to the system boundary described above (Fernandez et al., 2025; Patterson et al., 2021; ISO/IEC 21031:2024).

---

# 13. Stronger one-sentence formulation for Results / Discussion

Recommended:

> **Token reductions are treated as evidence of lower computational demand, while environmental implications are assessed separately through measured GPU operational energy; no proportional mapping from tokens to energy or CO₂e is assumed.**

Alternative if a grid factor is later added:

> **Operational GHG estimates are derived from measured GPU energy using an explicitly reported location- and time-specific electricity carbon-intensity factor, and therefore remain estimates within a device-level operational boundary rather than full lifecycle carbon footprints.**

---

# 14. Suggested Sustainability Methodology subsection

## Sustainability measurement

We separate computational efficiency from environmental impact. Token usage, latency and monetary inference cost are reported as measures of computational and economic efficiency and are not treated as direct proxies for greenhouse-gas emissions. This distinction is important because accelerator energy depends on hardware, serving configuration, software implementation and workload characteristics rather than token count alone (Fernandez et al., 2025).

Operational energy is instead measured directly from GPU telemetry and integrated over each inference run. We refer to this quantity as **GPU-device operational energy**. The measurement boundary excludes host CPU and memory, storage, networking and data-centre facility overhead, including cooling and power-distribution losses, because facility-specific PUE information was unavailable. Embodied emissions from hardware manufacture and infrastructure are also outside scope. Accordingly, the resulting measurements do not constitute total system energy, a full SCI score, or a lifecycle carbon footprint.

Where operational greenhouse-gas emissions are estimated, measured GPU energy is combined with an explicitly identified electricity carbon-intensity factor following the general SCI relation \(O=E\times I\), where \(E\) denotes electricity consumption and \(I\) grid carbon intensity (ISO/IEC 21031:2024). Carbon-intensity data should be matched to the physical execution region and, where timestamps permit, to the finest defensible temporal resolution. The resulting quantity is therefore interpreted as a **location-based GPU-device operational emissions estimate** within the stated system boundary.

---

# 15. Suggested Results wording

If token savings and GPU energy savings agree:

> The router reduced both token usage and measured GPU operational energy. The former establishes lower computational demand, while the latter provides direct evidence that the efficiency gain translated into lower accelerator energy consumption under the evaluated hardware and serving configuration. We do not assume that the percentage reduction in tokens should equal the percentage reduction in energy.

If token savings occur but GPU energy does not change materially:

> Although the router reduced token usage, this did not translate into a commensurate reduction in measured GPU operational energy. This divergence reinforces the distinction between computational workload proxies and physical energy consumption, which also depends on hardware utilisation, serving overheads and workload configuration.

If GPU energy was measured but CO₂e was not calculated:

> We deliberately stop the environmental accounting at measured GPU operational energy. Estimating operational CO₂e would additionally require a defensible electricity carbon-intensity factor matched to the execution time and location, while broader carbon-footprint claims would require a wider system boundary.

---

# 16. Suggested Limitations wording

> The environmental measurements in this study are intentionally bounded. GPU telemetry captures accelerator-side operational energy but omits non-GPU IT consumption, facility overhead and embodied hardware emissions. The results therefore permit comparisons of device-level operational energy between inference strategies under a controlled experimental configuration, but they should not be interpreted as complete carbon footprints or lifecycle environmental assessments. In addition, any mapping from operational energy to GHG emissions is deployment-dependent because electricity carbon intensity varies by location and time.

---

# 17. What not to write

Avoid:

> “The method uses 30% fewer tokens and therefore reduces carbon emissions by 30%.”

Reason:

- token count is not an energy measurement;
- energy/token varies with system configuration;
- emissions additionally require electricity carbon intensity;
- total environmental impact has a still wider boundary.

Avoid:

> “Our agent is 30% more sustainable.”

Reason:

- “sustainable” is multidimensional and undefined here;
- the experiment measures only specific efficiency and operational-energy metrics.

Avoid:

> “We measured the carbon footprint of inference.”

unless the measurement actually includes a defensible carbon-intensity conversion and the claimed system boundary is clearly defined.

Prefer:

> “We measured GPU-device operational energy during inference.”

or:

> “We estimate location-based GPU-device operational GHG emissions within the stated measurement boundary.”

---

# 18. Citation-ready core claim

A concise, defensible formulation is:

> **Computational efficiency and environmental impact are related but non-equivalent quantities: token count and FLOPs describe computational demand, whereas operational GHG emissions require an energy measurement or model and an explicit electricity carbon-intensity factor, with broader footprint claims additionally depending on the declared system boundary and embodied emissions.**

Suggested supporting citations:

- Fernandez et al. (2025);
- ISO/IEC 21031:2024;
- Lannelongue et al. (2021);
- Patterson et al. (2021).

---

# 19. Recommended minimum reporting checklist

For each experimental configuration, report where possible:

- model;
- accelerator model;
- accelerator count;
- inference precision;
- batch/concurrency configuration;
- serving software/runtime;
- number of runs/tasks;
- input tokens;
- output tokens;
- latency;
- monetary cost;
- GPU-energy measurement method;
- sampling interval;
- start/end timestamp;
- measured GPU energy in Wh/kWh;
- physical/cloud execution region;
- grid-carbon-intensity source, if used;
- temporal resolution of grid data;
- direct vs lifecycle grid factor;
- PUE included? yes/no;
- host CPU/memory included? yes/no;
- networking included? yes/no;
- embodied hardware emissions included? yes/no;
- functional unit used for normalisation;
- uncertainty / missing coverage.

A useful functional unit for an agent benchmark may be one of:

- per attempted task;
- per successful task;
- per benchmark episode;
- per correctly completed task.

Be explicit because the choice affects interpretation.

---

# 20. Recommended naming for figures and tables

Good:

- **Inference Cost and GPU Operational Energy**
- **Computational Efficiency and Device-Level Energy**
- **Token Usage, Latency, Cost, and GPU Energy**
- **GPU Operational Energy per Attempted Task**
- **GPU Operational Energy per Successful Task**

Avoid:

- **Environmental Impact**
- **Carbon Footprint**
- **Sustainability Score**

unless the plotted quantities genuinely support those broader labels.

---

# 21. UNcertain / unresolved items

## UNCERTAIN-1 — No universal “token-to-carbon” standard

No verified standard or canonical methodology establishes a universal conversion of:

\[
\text{tokens} \rightarrow \text{kWh}
\]

or:

\[
\text{tokens} \rightarrow \text{CO₂e}.
\]

The strongest defensible statement is therefore not:

> “Prior work proves tokens are an invalid carbon proxy.”

but:

> “Token count is a computational-demand metric and does not, by itself, determine either energy use or GHG emissions.”

This is supported structurally by SCI and empirically by inference-energy work such as Fernandez et al. (2025).

---

## UNCERTAIN-2 — Average versus marginal carbon intensity

There is no single carbon-intensity choice appropriate for all questions.

- attributional footprint → often average/location-based;
- causal avoided emissions → generally calls for marginal/consequential reasoning.

Therefore the dissertation should state which accounting question it answers.

---

## UNCERTAIN-3 — Exact semantics of third-party grid datasets

Different APIs may report:

- direct combustion emissions;
- lifecycle emissions;
- consumption-based intensity;
- production-based intensity;
- flow-traced intensity;
- forecast rather than actual values.

Before calculating final CO₂e, the exact semantic definition of the selected field should be recorded.

---

## UNCERTAIN-4 — PUE omission

There is no need to invent a PUE merely to produce a “complete-looking” number.

If facility-specific PUE is unavailable, a device-level measurement boundary is more transparent. A sensitivity analysis using plausible PUE values could be added if useful, but it should remain explicitly hypothetical rather than being mixed with directly measured values.

---

# 22. Practical recommendation for this dissertation

Given the current data, the strongest methodological position is:

> **Make GPU operational energy the environmental-relevant measured outcome, not token count.**

Use token count, latency and dollar cost as efficiency outcomes.

Then write:

> **Efficiency savings are environmentally relevant insofar as they reduce measured operational energy under the evaluated deployment conditions; they are not treated as direct or proportional measurements of GHG emissions.**

If precise execution timestamps and physical/cloud regions are available, optionally add:

\[
\text{measured GPU kWh}
\times
\text{time/location-specific grid CI}
=
\text{bounded operational CO₂e estimate}.
\]

If those are not available, **do not force a carbon number**. Device-level operational energy is already a legitimate sustainability-relevant measurement when its boundary is stated correctly.

---

# 23. Source index

1. **ISO/IEC 21031:2024 — Software Carbon Intensity specification**  
   https://www.iso.org/standard/86612.html  
   Public SCI specification: https://sci.greensoftware.foundation/

2. **ISO/IEC 30134-2:2026 — Power usage effectiveness (PUE)**  
   https://www.iso.org/standard/30134-2

3. **ITU-T L.1410 (11/2024) — Methodology for environmental life cycle assessments of ICT goods, networks and services**  
   https://www.itu.int/rec/T-REC-L.1410-202411-I/en

4. Fernandez et al. **Energy Considerations of Large Language Model Inference and Efficiency Optimizations**  
   DOI: `10.18653/v1/2025.acl-long.1563`  
   https://aclanthology.org/2025.acl-long.1563/

5. Lannelongue et al. **Green Algorithms: Quantifying the Carbon Footprint of Computation**  
   arXiv: `2007.07610`  
   DOI: `10.1002/advs.202100707`  
   https://arxiv.org/abs/2007.07610

6. Strubell et al. **Energy and Policy Considerations for Deep Learning in NLP**  
   arXiv: `1906.02243`  
   DOI: `10.18653/v1/P19-1355`  
   https://aclanthology.org/P19-1355/

7. Henderson et al. **Towards the Systematic Reporting of the Energy and Carbon Footprints of Machine Learning**  
   arXiv: `2002.05651`  
   https://arxiv.org/abs/2002.05651

8. Lannelongue et al. **GREENER principles for environmentally sustainable computational science**  
   DOI: `10.1038/s43588-023-00461-y`

9. Patterson et al. **Carbon Emissions and Large Neural Network Training**  
   arXiv: `2104.10350`  
   https://arxiv.org/abs/2104.10350

10. Luccioni et al. **Estimating the Carbon Footprint of BLOOM, a 176B Parameter Language Model**  
    arXiv: `2211.02001`  
    https://arxiv.org/abs/2211.02001

11. Gupta et al. **Chasing Carbon: The Elusive Environmental Footprint of Computing**  
    arXiv: `2011.02839`  
    DOI: `10.1109/MM.2022.3163226`

12. **NESO Carbon Intensity API**  
    https://api.carbonintensity.org.uk/  
    https://carbonintensity.org.uk/

13. **UK Government / DESNZ Greenhouse Gas Reporting Conversion Factors 2026**  
    https://www.gov.uk/government/publications/greenhouse-gas-reporting-conversion-factors-2026

14. **Electricity Maps API**  
    https://app.electricitymaps.com/developer-hub/api/reference

15. **Ember API**  
    https://api.ember-energy.org/v1/docs

16. **US EPA eGRID**  
    https://www.epa.gov/egrid/summary-data

17. **GHG Protocol Scope 2 Guidance**  
    https://ghgprotocol.org/scope-2-guidance

---

# 24. One-line takeaway

> **Report tokens as computational efficiency, measured GPU kWh as device-level operational energy, and only report CO₂e after an explicit time/location-specific electricity-carbon-intensity conversion; anything broader requires a wider declared system boundary.**
