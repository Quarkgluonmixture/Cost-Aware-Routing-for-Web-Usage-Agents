### One-line Verdict

**Needs restructure before OSF lock**: the core Hero + Structural + Framing-Rule design is good, but the current document is doing too many jobs, has stale 16-cell/router/mechanism material, and leaves H1 / heterogeneity / estimator logic too ambiguous for a DOI-cited preregistration.

### Structural Strengths

- **Hero vs structural split is the right epistemic shape.** §1 correctly avoids pretending P-text/P-prompt were deployment arms and frames them as ablations supporting phantom-space structure: [preregistration.md:32-38](/home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents/docs/checkpoints/pre_run/preregistration.md:32).

- **K-of-N demotion is conceptually correct.** Treating K as transparency rather than a gate is justified by the power file’s warning that per-cell power is weak at 1-5pp effects: [power_analysis.md:34-58](/home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents/docs/analysis/cross_sites/power_analysis.md:34).

- **Mode/cell/condition distinction is valuable.** The 24 operational conditions vs 4 statistical cells distinction is exactly the clarification this prereg needs: [preregistration.md:252-254](/home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents/docs/checkpoints/pre_run/preregistration.md:252).

- **Explicit post-hoc labeling is good.** H5/H6 are not falsely elevated to confirmatory status: [preregistration.md:122-134](/home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents/docs/checkpoints/pre_run/preregistration.md:122).

### Structural Problems

1. **H1 is over-specified and internally redundant.**  
   H1(i) requires pooled meta significance over 0; H1(ii) requires rejecting `H0: θ ≤ 1.0pp`. If the same pooled estimator/SE rejects superiority over +1pp, then it already implies the point estimate exceeds 1pp and the lower confidence bound is above +1pp, hence above 0. So H1(i) and the separate `θ_RE ≥ 1.0pp` magnitude check add no real decision information: [preregistration.md:52-55](/home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents/docs/checkpoints/pre_run/preregistration.md:52).  
   **Fix:** make H1 a single primary gate: “pooled effect significantly exceeds +1.0pp.” Report the >0 test as descriptive sanity check only, or delete it.

2. **R1-R5 reduces some forking paths but reintroduces discretion through the heterogeneity branch.**  
   The main R1-R5 table is useful, but the `I² > 75%` branch says “do NOT pool” and then switches to `≥3/4 direction-positive + ≥2 Holm sig → R3-grade hook`: [preregistration.md:179](/home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents/docs/checkpoints/pre_run/preregistration.md:179). That is a second, looser decision system not fully integrated with H1/H2/H3. It also conflicts with the superiority test, which needs a pooled estimate.  
   **Fix:** add a crisp decision flow: first heterogeneity check, then either pooled H1 test or declared “no pooled H1 inference.” If no pooling, define exactly whether H1 can pass, fail, or only support a downgraded descriptive claim.

3. **Router H7/H8 should not be in this prereg as full hypotheses.**  
   H7/H8 are pending advisor lock, need router data / Phase 2 design, and are not required for the workshop Phase 1a phantom-space claim: [preregistration.md:136-163](/home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents/docs/checkpoints/pre_run/preregistration.md:136). Keeping them creates a “preregistered but untested / silently deferred” liability. They also retain stale logic: H7 uses `TOST equivalence` wording and K gates after the reframe: [preregistration.md:142-144](/home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents/docs/checkpoints/pre_run/preregistration.md:142).  
   **Fix:** move H7/H8 to a separate “Future router preregistration stub / not part of this DOI claim” appendix, or remove them entirely from the OSF-locked document.

4. **DerSimonian-Laird as locked primary estimator is a weak choice at k=4.**  
   The doc locks DL as primary for N=4 cells: [preregistration.md:258](/home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents/docs/checkpoints/pre_run/preregistration.md:258). With k=4, DL τ² and CI behavior are fragile. A prereg can lock a known imperfect estimator, but it should not lock it as sole primary without advisor sign-off and sensitivity hierarchy.  
   **Fix:** make this an explicit advisor decision. Prefer primary `REML + Hartung-Knapp` or a clearly justified fixed/planned-cell average estimand, with DL retained as archive-compatible sensitivity.

5. **The 2026-05-13 reframe is incomplete.**  
   Stale examples: `16-cell rerun` in status text [preregistration.md:17](/home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents/docs/checkpoints/pre_run/preregistration.md:17), reading order still says H1-H6 [preregistration.md:19](/home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents/docs/checkpoints/pre_run/preregistration.md:19), H5 says tested against 16-cell data [preregistration.md:126](/home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents/docs/checkpoints/pre_run/preregistration.md:126), witness says 9 decisions then email says 8 [preregistration.md:320-333](/home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents/docs/checkpoints/pre_run/preregistration.md:320), OSF project/footnote still says 16-cell [preregistration.md:340-343](/home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents/docs/checkpoints/pre_run/preregistration.md:340). `followup.md` Part 2 is still old K-as-gate / 16-cell framing: [followup.md:57-87](/home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents/docs/checkpoints/followup.md:57).  
   **Fix:** do one scope pass before lock: Phase 1a = 24 conditions / 4 cells everywhere; router/mechanism/shop = deferred unless explicitly included.

6. **K_h1 and K_h3 are not distinct at N=4.**  
   `ceil(0.75*4)=3` and `ceil(0.67*4)=3`: [preregistration.md:238-239](/home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents/docs/checkpoints/pre_run/preregistration.md:238). Since they are transparency-only, two ratios imply a distinction that does not exist in this prereg.  
   **Fix:** replace with one row: “cell-consistency transparency count = 3/4 for H1 and H3.” Put the legacy 0.75/0.67 rationale in Appendix A only.

### Missing Content

- **A short power/sample-size paragraph inside the prereg itself.** The power appendix exists, but it is stale 16-cell/shop-inclusive in places: [power_analysis.md:5](/home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents/docs/analysis/cross_sites/power_analysis.md:5), [power_analysis.md:38-56](/home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents/docs/analysis/cross_sites/power_analysis.md:38). The prereg needs a 4-cell Phase 1a power limitation paragraph.

- **A crisp H1 pass/fail decision flow.** Especially: how `I² > 75%` affects H1, whether superiority can be tested without pooling, and whether R3 is a “pass,” “downgrade,” or “descriptive only.”

- **The estimand.** Is the target an inverse-variance weighted mean across four planned cells, an equal-cell average, or a random draw from a broader site/model population? This matters for DL/REML/fixed-effect choice.

- **Cost-equivalence decision rule clarity.** H2(a) says PRIMARY gate but also “replicated in ≥K_h2 = 3 of 4 cells (transparency consistency check)” [preregistration.md:77-79](/home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents/docs/checkpoints/pre_run/preregistration.md:77). Say whether cost passes by 3/4 cells, pooled equivalence, or by-construction token accounting.

### Over-Engineering / Scope Creep

- **Move H7/H8 router out.** Not needed for Phase 1a workshop prereg.

- **Move §5.X mechanistic layer-selection disclosure out.** It is detailed Stage 2 mechanism material, not part of the phantom-space Phase 1a prereg: [preregistration.md:279-312](/home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents/docs/checkpoints/pre_run/preregistration.md:279).

- **Compress operational protocol rows.** Stopping rules, contamination halts, failure-mode rubrics, and reproducibility machinery are useful, but §4 is overloaded: [preregistration.md:249-258](/home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents/docs/checkpoints/pre_run/preregistration.md:249). Keep a short “data validity protocol” section and link to an ops doc.

### Recommended Restructure

1. **Frontmatter + Scope**: Phase 1a only, 24 conditions / 4 statistical cells, what is excluded.  
2. **Epistemic Claim Hierarchy**: Hero, drop-in cost, structural axes, framing rule.  
3. **Primary Decision Flow**: H1/H2/H3 pass/fail, including heterogeneity branch.  
4. **Hypotheses and Families**: H1-H4 only for this DOI; H5/H6 as disclosure, not hypotheses.  
5. **Locked Analysis Choices**: only estimator, CI, bootstrap unit, FP filter, inclusion/exclusion, alpha, effect threshold.  
6. **Power and Limitations**: short 4-cell Phase 1a power note.  
7. **Witness / Reproducibility**: concise lock mechanics.  
8. **Appendices**: decision log, ops protocol link, router future prereg stub, mechanism disclosure.

Bottom line: the prereg has the right conceptual spine, but I would not OSF-lock this exact version. Strip it back to Phase 1a, make H1 one decisive test, resolve the no-pooling branch, and move router/mechanism/ops bulk out.