# Senior area-chair adjudication and final edit queue

**Manuscript:** *The Representation-Routing Gap for Web Agents*  
**Target:** REALM 2026, direct non-archival long paper  
**Inputs adjudicated:** manuscript PDF; cold-reader audit; scientific audit; external-evidence/citation audit; forensic numerical-consistency audit; venue-aware simulated review.  
**Decision standard:** an objection is retained only when it is directly supported by the manuscript, a primary source, or an official venue rule. Negative findings, method simplicity, lack of SOTA, and unspecified requests for baselines or experiments are not treated as defects.

---

## 1. Adjudicator’s bottom line

This is a viable measurement/negative-result paper, not a failed routing-method paper. The study has a coherent contribution after narrowing several absolute claims. It does **not** need a new agent run to become defensible. It does need one integrity pass through the prose and generated tables.

The five audits collapse to:

- **2 P0 submission blockers**: visible draft artefacts.
- **18 P1 edit clusters**: mostly exact factual contradictions, estimand naming, scope, and citation alignment. Several can be fixed by replacing one sentence or regenerating a table from an already-existing product.
- **10 P2 clusters**: useful improvements that can safely be deferred under the deadline.
- **8 DO NOT CHANGE rulings**: objections that would either damage the paper or demand work the claims do not require.

The paper’s defensible centre after these edits is:

> Under a fixed scaffold, the six tested end-to-end observation–grounding configurations yield nonidentical realised outcomes and behaviours. A same-condition rerun is a necessary comparator for attributing one-added-arm union gains. A retrospective outcome-informed oracle exposes a within-cell cost-saving opportunity, but the tested routing constructions do not provide a robust, correction-surviving route to it.

That is a meaningful REALM paper. REALM explicitly accepts non-archival eight-page long submissions and lists agent evaluation, architectures, robustness, realistic environments, and negative/preliminary work within scope.

---

## 2. Rules used to resolve disagreements

1. **A table beats a slogan.** When prose and the producing table disagree, the table/product is authoritative unless the table is itself internally inconsistent.
2. **Arithmetic validity and interpretation are separate.** A computed number can be correct while the name “ceiling,” “correction,” or “routing policy” is wrong.
3. **Observed-set claims are not expected-performance claims.** “Unchanged success by construction” is allowed only with “on the realised outcome matrix.”
4. **A finite set of failed algorithms is not a lower bound** unless a policy class and bound are formally defined.
5. **Citation omissions are prioritised by load-bearing relevance.** Direct, non-contemporaneous work that narrows the novelty or contradicts a headline claim is P1. Recent adjacent preprints do not justify rejection or new experiments.
6. **No new experiments are required** where a wording correction, existing sensitivity product, or existing task-by-mode matrix resolves the issue.

---

# 3. Disagreements between audits: final rulings

## D1. Is the “one wall” logical error a P0?

**Scientific audit:** “**Severity: P0 — Verified logical error**,” because the manuscript equates (i) tasks with at least one solver, (ii) tasks with more than one solver, and (iii) tasks with no solver.

**Venue and numerical audits:** reserve P0 for the visible title TODO; the numerical audit treats the label-supply generalisation as a major internal contradiction rather than a submission-format blocker.

**Source of disagreement:** severity semantics, not the underlying diagnosis.

**Ruling:** **CONFIRMED ERROR, P1—not P0.** It materially damages the central explanation, but it does not make the PDF unsubmitable in the way a visible drafting instruction does. Delete “the same set” and “one wall”; split which-mode label scarcity from binary triage/value failure.

## D2. Do all five routers fail to improve on fixed policies?

**Cold-reader audit:** “the five evaluated routers do not beat their relevant fixed baselines.”

**Scientific/venue/numerical audits:** Table 28 gives a red·B2 rule router at **5.42% / 0.08658 / 632.7s**, versus always-DOM at **3.94% / 0.09479 / 669.9s**; it improves all three displayed axes and adds a point to the fixed-mode frontier.

**Source of disagreement:** “relevant baseline” was silently changed from the entire fixed-mode frontier/best-success mode to the minimum-cost endpoint, always-cheapest.

**Ruling:** **CONFIRMED ERROR.** The absolute sentence is false. The stricter statement “no learned triage policy Pareto-dominates always-cheapest” can remain. The red·B2 rule result must be disclosed as a descriptive, leakage-sensitive exception—not promoted to a robust positive result.

## D3. Does the six-arm success ceiling “not survive” rerun correction?

**Manuscript/cold framing:** the headline ceiling “does not survive.”

**Scientific audit:** “The six-arm union remains correctly measured; only its attribution is unresolved.”

**Ruling:** **DEFENSIBLE REVIEWER CONCERN, resolved in favour of the scientific audit.** The realised six-arm union remains an oracle bound. What fails is the claim that its headroom is caused by representation diversity rather than extra-run opportunity.

## D4. Is the 9.5–30.6% number a valid result or an invalid “ceiling”?

**Numerical audit:** arithmetic is verified in 8/8 cells.

**Scientific/venue audits:** it is not the maximum lossless saving, is hindsight-labelled, and is not protected from future rerun instability.

**Ruling:** **Both are partly right.** The arithmetic is **confirmed**; the term “cost ceiling,” implementability language, and rerun-robust interpretation are **unsupported**. Rename it a “retrospective one-bit oracle saving” or “conservative oracle cost-saving opportunity.”

## D5. Should confirmed leaked successes remain in the primary estimates?

**Manuscript:** keeps them because the detector is a lower bound and reports zeroing as sensitivity.

**Venue/scientific audits:** confirmed invalid successes should be failures in the primary labels, even if the detector may miss more cases.

**Ruling:** **CONFIRMED METHODOLOGICAL ERROR.** Uncertainty about undetected leaks does not make six detected invalid completions valid. Corrected labels should be primary; raw benchmark-credit labels should be sensitivity. No new episodes are required.

## D6. Is appendix dependence a P1?

**Cold/venue audits:** central results are difficult to inspect without the appendix.

**Countervailing evidence:** the main paper already contains the main oracle table, and REALM permits unlimited appendix pages. The issue is presentation, not missing evidence.

**Ruling:** **DEFENSIBLE REVIEWER CONCERN, P2.** Add a compact policy summary only if space permits. Do not displace higher-priority factual fixes.

## D7. Should the provenance disclosure be removed or moved?

**Cold audit:** its chronology “damages confidence and consumes the main argument.”

**Venue review:** audit transparency is a strength.

**Ruling:** **OPTIONAL POLISH.** Keep the disclosure. Compress the chronology or move details to the appendix if space is needed, but do not conceal the defect ledger.

## D8. Are all missing 2026 papers “MUST CITE” before submission?

**External audit:** labels several 2026 works MUST CITE.

**ACL policy:** work appearing less than three months before the deadline is contemporaneous; authors are not obliged to perform detailed comparisons or new experiments and should receive benefit of the doubt for recent preprints.

**Ruling:** retain as P1 only the load-bearing, non-contemporaneous items: Avenir-Web; at least one repeated-execution reliability paper; BrowserGym as the established observation/action infrastructure; model primary sources; Benjamini–Hochberg; and the contrary grounding/planning decomposition. Region4Web and other very recent adjacent work are **P2 citation/context**, not novelty blockers and not experiment requirements.

---

# 4. Final edit queue

# P0 — must fix before upload

## P0.1 Remove visible draft artefacts

- **Status:** CONFIRMED ERROR.
- **Exact locations:** p.1 title block; p.6 between §§5 and 6.
- **Exact passages:**
  - “TODO working title — pick from the three candidates in the source”
  - “.*”
- **Exact edit:** delete both strings. Keep one final title. Recompile and visually inspect the rendered PDF.
- **Expected benefit:** removes the clearest desk-screening/reviewer-confidence failure.
- **Risk:** none.
- **Effort:** **tiny**.

---

# P1 — fix if at all possible

## P1.1 Define the treatment as an end-to-end observation–grounding configuration

- **Status:** CONFIRMED ERROR / estimand mismatch.
- **Locations:** Abstract; p.1 §1; p.3 §2; p.4 §3; Figure 1; Discussion.
- **Problematic passages:** “six representations”; “Holding this fixed is what makes a difference between conditions a difference of representation and not of scaffold”; “what representation changes is the first decision.”
- **Exact insertion at the start of §2.1:**

> “Throughout, a *mode* is an end-to-end observation–grounding configuration: it jointly specifies the observation payload, prompt family, identifier regime, and action-dispatch path. Because Vision uses coordinate dispatch while text-bearing modes use element IDs, the study estimates differences between deployable configurations, not the isolated causal effect of observation content.”

- **Exact global wording rule:** use “mode” or “observation–grounding configuration” in causal/result sentences. Retain “representation” only as informal shorthand after this definition.
- **Expected benefit:** blocks the strongest causal-validity objection without requiring a factorial experiment.
- **Risk:** slightly narrows the paper’s branding; does not weaken the practical deployment claim.
- **Effort:** **small**.

## P1.2 Standardise the six mode names

- **Status:** CONFIRMED ERROR.
- **Locations:** p.3 §2; all figures/tables; Appendix Table 50.
- **Problem:** the same image-free conditions are called DOM+stext, DOM+sprompt, SoM-image in the main paper and P-text, P-prompt, P-SoM in Table 50; “SoM-image” contains no image.
- **Exact edit:** use this grid everywhere:

| mode | text payload | prompt family | screenshot |
|---|---|---|---|
| DOM | accessibility tree | DOM | no |
| P-text | mark legend | DOM | no |
| P-prompt | accessibility tree | SoM | no |
| P-SoM | mark legend | SoM | no |
| SoM | mark legend | SoM | annotated |
| Vision | none | vision | unannotated |

- **Expected benefit:** makes every result traceable across main text and appendix.
- **Risk:** search-and-replace may miss plot labels; perform a PDF text search after regeneration.
- **Effort:** **small**.

## P1.3 Correct the unique-solve claim

- **Status:** CONFIRMED ERROR.
- **Locations:** p.1 Abstract/Introduction; p.4 §3; contribution summary.
- **Exact passage:** “Every mode solves tasks no other mode in its cell solves.”
- **Contradicting evidence:** Table 7 has four zero-unique mode–cell rows.
- **Exact replacement:**

> “Across the eight cells, all six modes solve tasks missed by other modes somewhere, and 44 of 48 mode–cell pairs have at least one unique realised success.”

- **Expected benefit:** removes a table-falsified absolute while preserving the non-redundancy result.
- **Risk:** “realised” is important; omitting it would again imply rerun-stable task complementarity.
- **Effort:** **tiny**.

## P1.4 Correct the 22% rerun headline

- **Status:** CONFIRMED ERROR.
- **Locations:** Abstract; p.1 §1; p.4–5 §4; p.7 §7; Table 11 caption.
- **Exact passage:** “Two runs of the same condition disagree on 22% of tasks.”
- **Problem:** 49/224 is the union of tasks flipping in at least one of two replicated conditions, not one condition’s disagreement rate. Per-condition discordance is 12.1–14.3%.
- **Exact replacement for Abstract/Introduction:**

> “In cls·B0, individual-condition reruns disagree on 12.1–14.3% of tasks; across the two replicated conditions used in the instability analysis, 49/224 tasks flip in at least one rerun pair.”

- **Exact replacement for Table 11 opening:**

> “Across two replicated cls·B0 conditions, 49/224 tasks changed outcome in at least one rerun pair; this is a two-condition union, not a single-condition discordance rate.”

- **Expected benefit:** repairs a headline numerical error and prevents an easy reviewer takedown.
- **Risk:** abstract length; shorten elsewhere rather than dropping the distinction.
- **Effort:** **tiny**.

## P1.5 Fix the rerun estimands and replication scope

- **Status:** CONFIRMED ERROR plus DEFENSIBLE REVIEWER CONCERN.
- **Locations:** p.4–6 §4–5; Figure 3; Tables 1, 23, 24; all “clears the floor/corrected” statements.
- **Required canonical vocabulary:**
  1. **observed mean rerun movement:** 0.89–2.23pp;
  2. **mean-difference sensitivity threshold:** 3.8–4.2pp;
  3. **one-sided set gain from a rerun:** 4.91–7.59pp in the three full cls·B0 pairs;
  4. **WA·B1 ten-task local draw:** report separately; do not merge it into a full-cell bound.
- **Exact replacement for Table 24/Figure 3 caption:**

> “Fusion premiums are mean success-rate differences and are compared with the 3.8–4.2pp mean-difference sensitivity threshold derived in §4. The 0.89–2.23pp observed shifts are descriptive draws, not the decision threshold. Outside the fully replicated cls·B0 cell, this comparison is a sensitivity analysis that assumes transfer of the threshold.”

- **Exact wording rule for unreplicated cells:** replace “clears/does not clear the rerun floor” with “would clear/would not clear if the cls·B0 threshold transferred.”
- **Exact Table 23 caption repair:** name both inventories: three full cls·B0 pairs and five WA·B1 modes on a registered ten-task draw.
- **Expected benefit:** protects the paper’s central methodological contribution from internal contradiction.
- **Risk:** makes several all-cell conclusions conditional; that is the correct scope.
- **Effort:** **small**.

## P1.6 Preserve the six-arm union, narrow its attribution

- **Status:** DEFENSIBLE REVIEWER CONCERN; current wording overstated.
- **Locations:** Abstract; p.1 §1; p.6 §5; p.8 Discussion.
- **Problematic passages:** “The headline ceiling does not survive that control”; “only one survives.”
- **Exact replacement:**

> “The realised six-arm union remains an observed oracle bound, but the available reruns show that its headroom cannot be attributed uniquely to representation diversity: in cls·B0, a rerun of an existing arm yields a one-sided set gain comparable to adding a distinct arm.”

- **Expected benefit:** keeps the valid measurement while removing a logically incorrect “disappears” claim.
- **Risk:** the rhetorical contrast becomes less dramatic.
- **Effort:** **tiny**.

## P1.7 Rename and scope the cost result

- **Status:** CONFIRMED TERMINOLOGY ERROR plus DEFENSIBLE REVIEWER CONCERN.
- **Locations:** Abstract; p.1 §1; p.6 §5; Table 1; p.7–8 §§7–9.
- **Problematic terms:** “cost ceiling”; “routing only the never-solved tasks”; “unchanged success”; “reachable with one bit per task”; “survives the rerun objection.”
- **Exact replacement for the headline sentence:**

> “A retrospective one-bit oracle that identifies tasks unsolved by all six realised runs assigns only those tasks to the cheapest mode, reducing the within-cell per-attempt cost proxy by 9.5–30.6% while preserving the best fixed mode’s realised solved set.”

- **Exact follow-up sentence:**

> “This construction is not subject to the extra-arm confound, but it is outcome-informed, is not an implementable policy, is not the maximum possible lossless saving, and does not establish unchanged expected success on a future rerun.”

- **Global rename:** “retrospective one-bit oracle saving” or “conservative oracle cost-saving opportunity.”
- **Expected benefit:** retains the strongest cost result in defensible form.
- **Risk:** “cost proxy” is less punchy; necessary because B0 is an API bill and B1/B2 are electricity estimates.
- **Effort:** **small**.

## P1.8 Replace “lower bound” with a finite stress test

- **Status:** CONFIRMED TERMINOLOGY ERROR.
- **Locations:** p.2 §1; contribution list; p.6 heading; p.7 §7; Table 35 captions.
- **Exact replacements:**
  - Heading: **“Five routing constructions and their observed failure modes”**.
  - Contribution: **“A stress test of five concrete routing constructions, with distinct observed obstructions.”**
  - Replace “lower bound” with “evaluated routing baseline set” or “finite construction suite.”
- **Expected benefit:** eliminates an indefensible theoretical claim while preserving the negative-result contribution.
- **Risk:** none scientifically; only rhetorical.
- **Effort:** **tiny**.

## P1.9 Correct the “none improves on fixed policies” claim

- **Status:** CONFIRMED ERROR.
- **Locations:** Abstract; p.2 §1; p.6 §6; p.8 §7; Figure 1 caption.
- **Exact replacement for Abstract:**

> “The tested constructions do not yield a robust, correction-surviving improvement over fixed policies. No learned triage policy Pareto-dominates always-cheapest under the strict success–cost criterion; before leakage correction, a red·B2 zero-token rule adds a descriptive point to the fixed-mode frontier.”

- **Exact replacement in §6:**

> “The conclusions are comparator-specific: the nested triage does not Pareto-dominate always-cheapest, and the cascade does not exceed always-rich. The red·B2 rule router descriptively dominates always-DOM on the displayed axes, but the cell is sparse and leakage-sensitive.”

- **Delete:** “Every implementable policy we could construct sits at or below the trivial fixed policies”; “reached by none of the five policies.”
- **Expected benefit:** reconciles the abstract with Table 28 without turning the paper into a positive-method claim.
- **Risk:** the red·B2 result may change when corrected leakage labels are made primary; phrase it explicitly as pre-correction until regeneration.
- **Effort:** **small**.

## P1.10 Split which-mode label scarcity from binary triage failure

- **Status:** CONFIRMED LOGICAL ERROR.
- **Locations:** Abstract; p.2 §1; p.6 §6; p.7 §7; Table 20 title/caption.
- **Delete:** “the labels a router can learn from and the tasks where routing can pay are the same set”; “one wall”; unqualified “routing supervision is produced at the success rate.”
- **Exact replacement:**

> “Six-way successful-mode supervision is available only on tasks solved by at least one mode and is sparse in these cells. Binary solvability triage is labelled on every task, but the tested nested policy fails for a different reason: predictability does not translate into a Pareto improvement over always-cheapest.”

- **Replace the reversal prediction with:**

> “Across eight correlated cells, the observed association between best-mode success and multi-solver share (ρ=0.952) motivates the hypothesis that successful-mode label supply may improve in higher-success regimes; it is not evidence of a guaranteed reversal.”

- **Rename Table 20:** “Association between success, label supply, and multi-solver share.”
- **Expected benefit:** repairs the paper’s central causal story without discarding either negative result.
- **Risk:** removes the memorable “one wall” slogan.
- **Effort:** **small**.

## P1.11 Fix the triage AUROC inconsistencies

- **Status:** CONFIRMED ERROR / CANNOT VERIFY which product is authoritative.
- **Locations:** Figure 1 caption; p.6 §6; Appendix Tables 19 and 31.
- **Conflicting numbers:** 0.53; 0.483; stated range 0.651–0.717; Table 31 baselines differ again.
- **Exact minimum edit if Table 19 is authoritative:**

> “The triage label is defined for every task; cross-validated AUROC spans 0.483–0.717 across the six VWA cells, with red·B2 below chance at 0.483.”

- **Required table action:** regenerate Figure 1, §6, Tables 19 and 31 from one authoritative product. If Table 31 uses a different feature set, fold assignment, or cohort, state that difference in its caption; otherwise delete its conflicting baseline column.
- **Expected benefit:** removes a conspicuous three-way numerical contradiction.
- **Risk:** none; do not guess which unseen product is correct.
- **Effort:** **small**.

## P1.12 Remove the false pooling licence

- **Status:** CONFIRMED ERROR.
- **Locations:** Appendix Tables 3 and 6; p.3 §2.
- **Contradiction:** Table 3 says grouping is “licensed by the non-separability result”; Table 6 says “This is not a separability test.”
- **Exact replacement for Table 3 caption:**

> “We use the four image-free modes as a deployment category for reporting convenience. The repeated-extrema tally in Table 6 does not establish statistical equivalence or license pooling.”

- **Rename Table 6:** “Repeated-extrema tally among image-free modes.”
- **Expected benefit:** removes a direct appendix self-contradiction.
- **Risk:** none; the descriptive class comparison can remain.
- **Effort:** **tiny**.

## P1.13 Narrow the winner-reversal interpretation

- **Status:** DEFENSIBLE REVIEWER CONCERN.
- **Locations:** Abstract; p.1 §1; p.4 §3; p.8–9 Discussion/Limitations.
- **Problematic passage:** “the winner is a property of the deployment and not of the modality.”
- **Exact replacement:**

> “The observed arm-matched ordering differs between the classifieds and WA-reddit task sets in these cells. Because task set, application state, and backbone coverage co-vary, the moderator of that reversal is not identified.”

- **Expected benefit:** preserves the deployment-specific empirical message without claiming a universal causal moderator.
- **Risk:** none.
- **Effort:** **tiny**.

## P1.14 Make confirmed leakage correction primary

- **Status:** CONFIRMED METHODOLOGICAL ERROR.
- **Locations:** Table 1 caption; p.8 §8; Tables 48–49; all red/B2 dependent conclusions.
- **Exact analysis edit:** set the six confirmed leaked VWA successes to failure in the primary outcome product; keep the original benchmark-credit labels as a sensitivity analysis.
- **Exact prose replacement:**

> “Primary analyses treat the six hand-confirmed environmentally credited episodes as failures. The original benchmark-credit labels are reported as a sensitivity analysis; the detector is incomplete, so the corrected estimates may remain optimistic.”

- **Regenerate from the corrected product:** Tables 1–7, 20–35, 48–49 and all red/B2 routing/winner statements that depend on those outcomes.
- **Caption arithmetic repair:** VWA has 6 leaked and **31 earned** among 37 displayed successes; the combined VWA+WA total is 6 leaked and **68 earned**. Point Table 49 to Table 48, not Table 28.
- **Expected benefit:** prevents the paper from knowingly using invalid completions as its primary scientific labels.
- **Risk:** may weaken or remove the red·B2 exception and one contrast; this is a necessary correction, not an optional robustness choice.
- **Effort:** **substantial** relative to other edits, but no new agent runs.

## P1.15 Rename the six-way label or regenerate it from measured cost

- **Status:** CONFIRMED DEFINITION ERROR.
- **Locations:** p.6 §6; Tables 25–26; Appendix §I.2/Table 51.
- **Problematic passage:** “The natural label, the cheapest mode that solved the task.”
- **Evidence:** the implementation uses a fixed priority order and selects a strictly pricier successful mode on 12.5–54.6% of labelled tasks.
- **Minimum exact replacement:**

> “The six-way target is the first successful mode under a fixed priority order; it is not the measured cheapest successful mode and is therefore analysed as a successful-mode label rather than a cost-optimal routing label.”

- **Alternative stronger fix using existing data:** regenerate the label from the task-by-mode solve matrix and measured within-cell mode cost, then rerun the existing classifier analysis.
- **Expected benefit:** aligns the supervised target with what the code actually predicts.
- **Risk:** the stronger alternative may alter results; the minimum rename is safe but narrows cost interpretation.
- **Effort:** **tiny** for rename; **substantial** for relabelling/reanalysis.

## P1.16 Repair the closest-work and novelty paragraph

- **Status:** DEFENSIBLE REVIEWER CONCERN; broad novelty wording is unsupported.
- **Locations:** first two Introduction paragraphs; routing-related work paragraph.
- **Required insertion:**

> “Most benchmarked systems still commit globally to an observation/grounding interface, although recent work has begun to adapt grounding or observation content conditionally. Avenir-Web, for example, combines visual grounding with a structural fallback inside a richer planning-and-memory system. Our claim is therefore not the first conditional grounding selector: we isolate task-level choice among six explicit end-to-end modes under a fixed scaffold and evaluate it against same-condition reruns.”

- **Also cite/position:** BrowserGym as the established shared observation/action infrastructure; at least one repeated-execution reliability study; Adaptive VLM Routing/WebRouter as model-routing rather than representation-routing.
- **Contemporaneous-work ruling:** Region4Web, FocusAgent/LineRetriever, A11y-Compressor, and very recent benchmark-audit preprints should be cited briefly if space permits, but no new empirical comparison is required.
- **Expected benefit:** prevents a novelty objection while sharpening what is genuinely new.
- **Risk:** none; the controlled measurement/rerun contribution remains distinct.
- **Effort:** **small**.

## P1.17 Correct load-bearing citation claims

- **Status:** mixture of CONFIRMED ERROR and DEFENSIBLE REVIEWER CONCERN.
- **Exact edits:**
  1. **Mind2Web:** remove it from the sentence claiming online agents make a fixed text/pixels/SoM choice at build time; retain it as dataset/background.
  2. **Grounding bottleneck:** replace “the bottleneck” with “a major bottleneck under that setup,” and acknowledge the contrary planning-dominant decomposition from *From Grounding to Planning*.
  3. **WebArena audits:** replace “both benchmarks have been re-audited” with “WebArena has been re-audited by WebArena Verified; our VisualWebArena leakage audit is internal,” unless a genuine VWA audit source is added.
  4. **Gupta cascade:** replace “built in the manner of” with “motivated by language-model cascading, we test a simpler free-form self-reported-confidence cascade.”
  5. **Kadavath:** call the signal “free-form self-reported confidence” and state that it differs from P(True)/P(IK)-style elicitation.
  6. **Hajimiri:** replace “erases the gains” with “substantially narrows, and in aggregate can remove, reported gains, with domain-specific exceptions.”
  7. **Xue citation:** remove or replace it unless its primary source actually audits the benchmark named by the sentence.
- **Expected benefit:** removes several sentence-level citation mismatches reviewers can verify quickly.
- **Risk:** none.
- **Effort:** **small**.

## P1.18 Repair statistical attribution, model provenance, endpoint causality, and internal references

- **Status:** CONFIRMED ERRORS plus CANNOT VERIFY.
- **Locations:** p.3 multiplicity paragraph; p.3 model setup; p.5 nondeterminism explanation; Appendix captions/references.
- **Exact edits:**
  - Add Benjamini & Hochberg (1995) for BH. State that `max(p1,p2)` is the paper’s conjunction/intersection–union construction; Holm (1979) supports Holm correction only.
  - Add primary Qwen3-VL and Gemma 3 technical-report/model-card citations and exact checkpoint/revision identifiers.
  - Replace the unverified hosted-MoE causal sentence with:

> “B0 is served by a hosted endpoint whose server-side execution is not exposed. Its non-bit-reproducibility may reflect inference-service behaviour and/or environment drift; these components cannot be separated from the available logs.”

  - Delete the universal sentence “Every stability figure in this literature is a single run.” Replace it with:

> “Single-run reporting remains common, but repeated-execution studies already document substantial agent unreliability; our contribution is to use a same-condition rerun comparator for page-observation comparisons in web agents.”

  - Integrity sweep: Table 35 must summarise the actual five constructions or be relabelled; Table 48/49 cross-references and “WA unaudited” chronology must agree; fix `cla·SoM`; reconcile `last k` with “last eight”; remove stale section/table numbers.
- **Expected benefit:** improves reproducibility and prevents easy factual/citation objections.
- **Risk:** none.
- **Effort:** **small**.

---

# P2 — valuable but safe to defer

## P2.1 Early-divergence interpretation

- **Status:** DEFENSIBLE REVIEWER CONCERN.
- **Passage:** “what representation changes is the first decision.”
- **Minimum fix:** “On outcome-disagreement tasks, mode assignments often produce different early trajectories.”
- **Why deferable:** the appendix already labels the selected task cohort; this is interpretation, not a numerical contradiction.

## P2.2 Main-paper self-containment and figures

- **Status:** DEFENSIBLE REVIEWER CONCERN / OPTIONAL POLISH.
- **Items:** move the six-mode grid into §2; add a compact five-policy outcome box; make Figure 1 show failed/exceptional policies; de-emphasise unstable winner circles in Figure 2.
- **Ruling:** useful under an eight-page paper, but do not sacrifice P0/P1 fixes. Appendix use is permitted.

## P2.3 Outcome-informed task exclusion

- **Status:** DEFENSIBLE REVIEWER CONCERN.
- **Fix:** label the second exclusion explicitly as post hoc; make unfiltered results primary if any headline verdict changes, otherwise state that all headline conclusions are unchanged with both tasks restored.

## P2.4 Conditional-rate denominators and incomplete rows

- **Status:** CONFIRMED DATA-PRESENTATION ERROR.
- **Items:** undefined click/type failure rates encoded as 0; Table 41’s cls·B0 SoM denominator 187 rather than 224; pooled-label universes change without a map.
- **Fix:** report complete-case conditional rates plus zero-denominator fraction; explain or repair the 187-row cohort; add a one-line universe map to pooled analyses.

## P2.5 Inferential-language hygiene

- **Status:** DEFENSIBLE REVIEWER CONCERN.
- **Items:** “significant” inside families declared descriptive; task-bootstrap intervals conditional on realised runs; post-outcome 7/8 threshold in Table 6.
- **Fix:** use “interval excludes zero” rather than generic “significant”; add “conditional on the realised runs” to bootstrap captions; keep Table 6 explicitly descriptive and do not use it to license equivalence.

## P2.6 Small numerical/prose corrections

- **Status:** CONFIRMED ERROR.
- **Items and exact fixes:**
  - “precisely two latency exceptions” → “three exceptions: two latency metrics and parse-fail rate.”
  - Figure 4 “null on reddit” → “no positive flagged-task advantage on reddit; several estimates are negative.”
  - “same money” → “same arm count.”
  - Table 29 “rich is worse” for equality → “cheap is not worse / premise not satisfied.”
  - “always-Vision” claim → scope it to the exact cells/policies shown in Table 28.

## P2.7 Provenance placement, captions, and prose compression

- **Status:** OPTIONAL POLISH.
- **Ruling:** keep the provenance disclosure and limitations. Compress chronology, move product filenames/details to appendix, shorten captions that currently contain argumentation, and reduce repeated antitheses/slogans.

## P2.8 Raw paths and minor build artefacts

- **Status:** OPTIONAL POLISH / CONFIRMED BUILD ERROR.
- **Fix:** replace internal filesystem paths with semantic dataset descriptions; reconcile Figure 1’s `k` with exactly eight history items; set PDF metadata title; run a final cross-reference and glyph search.

## P2.9 Bibliography metadata cleanup

- **Status:** CONFIRMED METADATA ISSUES, non-load-bearing.
- **Items:** complete UGround authors; cite Gupta as ICLR 2024; correct RouteLLM title; cite He as sole author; use archival Yuan version; correct VisualWebArena title/record; remove unsupported parentheticals from WebArena Verified.

## P2.10 Adjacent recent literature

- **Status:** USEFUL CONTEXT, not a blocker.
- **Candidates:** Region4Web; FocusAgent/LineRetriever; A11y-Compressor; DiMo-GUI; WAREX; WebGym; recent benchmark mis-scoring work; randomness studies outside web agents.
- **Ruling:** cite compactly where feasible. Do not add experiments or surrender novelty solely because these works exist.

---

# DO NOT CHANGE — suggestions likely to damage or unnecessarily weaken the paper

1. **Do not add a SOTA baseline merely to make the numbers look competitive.** The paper’s contribution is measurement, controls, and a negative result.
2. **Do not hide the negative result or rewrite the paper as a new routing algorithm paper.** That would invite a much harsher standard and misstate the work.
3. **Do not delete the rerun analysis, leakage disclosure, defect ledger, or limitations.** These are among the paper’s strongest contributions; correct their wording instead.
4. **Do not request a factorial representation-only experiment** after the treatment is correctly named an end-to-end configuration. Such an experiment is necessary only if the paper insists on a modality/content causal claim.
5. **Do not run full-cell reruns in all six unreplicated cells** unless retaining unconditional eight-cell “rerun-corrected” claims. Conditional sensitivity wording is sufficient for this submission.
6. **Do not add unspecified “missing baselines.”** A reviewer must name a baseline and the claim it tests. Model-routing systems are related work, not automatically mandatory experimental comparators.
7. **Do not call the paper non-novel because Avenir-Web or model routers exist.** They narrow the novelty to controlled task-level observation–grounding measurement plus rerun-matched ceilings; they do not pre-empt that contribution.
8. **Do not drop all B2 cells.** Keep them as sparse-regime coverage, but do not use red·B2 as confirmatory evidence until leakage-corrected and clearly labelled descriptive.

---

# 5. Complete disposition map for the five audits

Legend: **CE** confirmed error; **DRC** defensible reviewer concern; **IO** invalid/unsupported objection; **DUP** duplicate of another canonical item; **OP** optional polish; **CV** cannot verify from supplied evidence.

## 5.1 Cold-reader audit

| Original issue | Final disposition | Queue mapping | Adjudication |
|---|---|---|---|
| 1. Visible draft artefacts | CE | P0.1 | Confirmed, including TODO and `.*`. |
| 2. Incompatible rerun thresholds | CE | P1.5 | Confirmed; three estimands must be separated. |
| 3. AUROC inconsistency | CE/CV | P1.11 | Confirmed conflict; authoritative product cannot be inferred. |
| 4. Mode names | CE | P1.2 | Confirmed. |
| 5. Grouping licensed/unlicensed | CE | P1.12 | Confirmed. |
| 6. Representation conflates dispatch | CE | P1.1 | Confirmed estimand mismatch. |
| 7. Cost oracle sounds implementable | DRC | P1.7 | Valid; arithmetic retained, name narrowed. |
| 8. Rerun coverage over-scoped | DRC | P1.5 | Valid. The audit’s suggested “21.9% for one condition” is itself too broad; use P1.4 wording. |
| 9. Main paper outsources evidence | DRC | P2.2 | Downgraded from P1; appendix is allowed and evidence exists. |
| 10. Lower bound overstated | CE | P1.8 | Confirmed. |
| 11. Provenance placement damages confidence | OP | P2.7 / DO NOT CHANGE 3 | Compress if needed; do not remove. |
| 12. Figure 1 not showing result | OP | P2.2 | Valid design feedback, not scientific blocker. |
| 13. Figure 2 endorses unstable winners | DRC | P2.2 | Valid presentation concern. |
| 14. Captions carry arguments | OP | P2.7 | Optional. |
| 15. Theatrical prose | OP | P2.7 | Style preference with some readability value. |

## 5.2 Scientific audit

| Original issue | Final disposition | Queue mapping | Adjudication |
|---|---|---|---|
| 1. “One wall” conflates three sets | CE | P1.10 | Confirmed, downgraded from P0 to P1. |
| 2. End-to-end treatment | CE | P1.1 | Confirmed. |
| 3. Every mode false | CE | P1.3 | Confirmed. |
| 4. Early divergence ≠ stable complementarity | DRC | P2.1 | Valid scope concern; not a fatal flaw after “realised” wording. |
| 5. Grouping contradiction | DUP/CE | P1.12 | Duplicate, confirmed. |
| 6. Winner reversal overgeneralised | DRC | P1.13 | Valid. |
| 7. Full-cell rerun only once | DRC | P1.5 | Valid. |
| 8. Success ceiling survives as estimate | DRC | P1.6 | Better interpretation adopted. |
| 9. Inconsistent fusion bands | DUP/CE | P1.5 | Duplicate, confirmed. |
| 10. Not a cost ceiling | CE | P1.7 | Terminology confirmed; number retained. |
| 11. Unchanged success only realised | DRC | P1.7 | Valid. |
| 12. Mixed cost proxies | DRC | P1.7 | Valid; use within-cell per-attempt cost proxy. |
| 13. Five constructions not lower bound | CE | P1.8 | Confirmed. |
| 14. Router counterexample | CE | P1.9 | Confirmed by Table 28. |
| 15. “Cheapest successful” label mismatch | CE | P1.15 | Confirmed. |
| 16. Floors imported into six cells | DUP/DRC | P1.5 | Duplicate, valid. |
| 17. Corrected success in sparse B2 cell | DRC | P1.9/P1.14 | Valid reason to keep exception descriptive. |
| 18. “Significance” in descriptive family | DRC | P2.5 | Wording fix; not P1. |
| 19. Bootstrap omits run uncertainty | DRC | P2.5 | Valid; caption caveat sufficient. |
| 20. Table 6 post-outcome threshold | DRC | P2.5/P1.12 | Remove licensing claim; descriptive table may remain. |
| 21. Leakage not propagated | CE | P1.14 | Confirmed. |

## 5.3 Venue-aware simulated review

| Original issue | Final disposition | Queue mapping | Adjudication |
|---|---|---|---|
| P0 title TODO | CE | P0.1 | Confirmed. |
| M1 no router improves | CE | P1.9 | Confirmed. |
| M2 every mode unique | DUP/CE | P1.3 | Confirmed. |
| M3 22% conflation | CE | P1.4 | Confirmed. |
| M4 transported threshold | DRC | P1.5 | Valid. |
| M5 cost ceiling | DUP/DRC | P1.7 | Valid. |
| M6 leaks retained primary | CE | P1.14 | Confirmed. |
| M7 lower bound not self-contained | DRC | P1.8 + P2.2 | Terminology is P1; self-containment is P2. |
| m1 grouping caption | DUP/CE | P1.12 | Confirmed. |
| m2 AUROC range | DUP/CE | P1.11 | Confirmed. |
| m3 Table 41 n=187 | CE | P2.4 | Confirmed but non-headline. |
| m4 outcome-informed exclusion | DRC | P2.3 | Valid, substantially addressed by sensitivity. |
| m5 undefined rates as perfect | CE | P2.4 | Confirmed. |
| m6 build/cross-reference artefacts | CE/OP | P0.1, P1.18, P2.8 | Split by severity. |
| “lack of SOTA” objection | IO | DO NOT CHANGE 1 | Explicitly rejected. |
| “negative result” objection | IO | DO NOT CHANGE 2 | Explicitly rejected. |
| unspecified novelty/baseline demands | IO | DO NOT CHANGE 6–7 | Explicitly rejected. |

## 5.4 Numerical/internal-consistency audit

| ID | Final disposition | Queue mapping | Note |
|---|---|---|---|
| I01 title instruction | CE | P0.1 | Confirmed. |
| I02 unique solves | CE/DUP | P1.3 | Confirmed. |
| I03 22% union | CE | P1.4 | Confirmed. |
| I04 comparator/scope | CE/DRC | P1.5 | Confirmed. |
| I05 fusion quantities | CE | P1.5 | Confirmed. |
| I06 AUROC 0.483/0.53/range | CE | P1.11 | Confirmed. |
| I07 Tables 19/31 | CV | P1.11 | Conflict confirmed; cause cannot be verified. |
| I08 label-supply generalisation | CE | P1.10 | Confirmed. |
| I09 >1 solver not only routing set | CE | P1.10 | Confirmed. |
| I10 no-router claim | CE | P1.9 | Confirmed. |
| I11 pooling licence | CE | P1.12 | Confirmed. |
| I12 binary evaluator only VWA | CE | P1.18/P2.4 | Scope any “whole paper” statement to VWA; small edit. |
| I13 winner reversal underspecified | DRC | P1.13 | Valid. |
| I14 Table 35 set mismatch | CE | P1.18 | Confirmed. |
| I15 stale references | CE | P1.18 | Confirmed integrity issue. |
| I16 Table 49 earned count | CE | P1.14 | Confirmed arithmetic. |
| I17 Table 48/49 chronology | CE | P1.18 | Confirmed stale caption. |
| I18 wrong sensitivity table | CE | P1.14/P1.18 | Confirmed. |
| I19 three behavioural exceptions | CE | P2.6 | Confirmed, low impact. |
| I20 “null on reddit” | CE | P2.6 | Confirmed wording error. |
| I21 “same money” | CE | P2.6 | Confirmed; arm count is matched, not cost. |
| I22 mode names | CE | P1.2 | Confirmed. |
| I23 pooled denominators | DRC | P2.4 | Valid documentation issue. |
| I24 Table 41 n=187 | CE/CV | P2.4 | Denominator mismatch confirmed; cause cannot be verified. |
| I25 Table 23 omits WA local draw | CE | P1.5 | Confirmed. |
| I26 equality called “rich worse” | CE | P2.6 | Confirmed. |
| I27 always-Vision not cell-general | CE | P2.6/P1.9 | Confirmed; scope to cells. |
| I28 `cla·SoM` | CE | P1.18 | Typo. |
| I29 k versus eight | CE | P2.8 | Minor consistency. |
| I30 raw paths | OP | P2.8 | Replace for professionalism/anonymity hygiene. |

---

# 6. External citation audit: adjudicated disposition

## 6.1 Existing citations

| Cited work | External-audit objection | Final category | Final action |
|---|---|---|---|
| Cawley & Talbot (2010) | Citation supports selection bias, not categorical “cannot be effect estimate.” | DRC, P2 | Split cited selection-bias claim from the paper’s own estimand statement. |
| FrugalGPT | Exact for routing; universal capability gradient too broad. | DRC, P2 | “Typically exploit an average capability–cost gradient.” |
| Mind2Web | Does not support fixed online observation-interface claim. | CE, P1 | Remove from that sentence. |
| WebArena Verified | Supports WebArena only, not “both benchmarks.” | CE, P1 | Name WebArena; internal VWA audit separately. |
| UGround | Substantively aligned; incomplete bibliography. | OP/CE metadata, P2 | Complete archival authors/record. |
| Language Model Cascades | Method differs from free-form confidence scalar. | CE, P1 | “Motivated by”; do not imply method replication. |
| Hajimiri et al. | “Erases gains” too absolute. | CE, P1 | Use narrowed aggregate-with-exceptions wording. |
| He (2025) | Alignment exact; publisher treated as coauthor. | CE metadata, P2 | Cite Horace He alone. |
| Holm (1979) | Does not support BH or automatically justify max-p conjunction. | CE, P1 | Add Benjamini–Hochberg; explain conjunction construction. |
| Kadavath et al. | Broader confidence evidence, not same elicitation. | DRC, P2 | Describe signal and protocol difference. |
| VisualWebArena | Benchmark identity exact; fixed-choice generalisation broader than source. | DRC/metadata, P2 | Narrow sentence; correct record. |
| WebRouter | Supports web-agent model routing, not universal monotonic gradient. | DRC, P2 | Distinguish routed object. |
| Adaptive VLM Routing | Highly relevant routing context. | DRC, P1 | Explicitly distinguish model routing from representation routing. |
| AgentRewardBench | Substantive alignment exact; version polish. | OP, P2 | Use best archival/official record. |
| Moslem & Kelleher survey | No substantive error. | IO/OP, P3 | No required change. |
| RouteLLM | Routing support exact; title metadata wrong. | CE metadata, P2 | Correct title; narrow universal gradient. |
| Sclar et al. | Supports formatting sensitivity with scope caveat. | DRC, P2 | Replace “carry no information” with “semantically equivalent/meaning-preserving.” |
| Xue et al. | Does not support the stated benchmark re-audit claim. | CE, P1 | Remove or replace with correct primary source. |
| Set-of-Mark Prompting | Citation relevant; manuscript description oversimplifies mechanism. | DRC, P2 | Describe mark overlay/legend accurately. |
| Yuan et al. | Alignment exact; better archival version available. | CE metadata, P2 | Cite archival version. |
| SeeAct | Supports grounding importance, not unqualified “the bottleneck.” | DRC, P1 cluster | Scope and cite contrary decomposition. |
| WebArena | Benchmark identity exact; fixed-choice claim is broader. | DRC, P2 | Narrow claim to evaluated agents/configurations. |

## 6.2 Missing-work recommendations

| Work/group | External audit | Final ruling |
|---|---|---|
| BrowserGym ecosystem | MUST CITE | **P1 citation.** It is established infrastructure directly framing the observation/action design space; no new experiment. |
| Avenir-Web | MUST CITE | **P1 citation and novelty distinction.** Direct conditional grounding overlap, older than contemporaneous window. |
| On the Reliability of Computer Use Agents | MUST CITE | **P1 citation.** Directly invalidates the universal single-run-literature sentence. |
| Benjamini & Hochberg (1995) | MUST CITE | **P1.** Required for the named procedure. |
| Qwen3-VL / Gemma 3 primary sources | MUST CITE | **P1.** Required for model provenance/reproducibility. |
| From Grounding to Planning | MUST CITE | **P1.** Material contrary evidence to “the bottleneck.” |
| Does the Way You Plan Matter? | MUST CITE | **P2/P1 only if the universal novelty sentence remains.** Cite briefly; no detailed comparison required if contemporaneous under the exact release date. |
| Region4Web | MUST CITE | **P2.** Very recent/contemporaneous observation-space work; brief discussion feasible, no experiment. |
| FocusAgent/LineRetriever | SHOULD CITE | **P2.** Task-conditioned observation adaptation; useful closest context, not a blocker. |
| A11y-Compressor | SHOULD CITE | **P2.** Representation/cost context. |
| DiMo-GUI | SHOULD CITE | **P2.** Dynamic visual grounding context. |
| WAREX | SHOULD CITE | **P2.** Environment-instability context, not identical-rerun evidence. |
| On Randomness in Agentic Evals | SHOULD CITE | **P2.** Corroborative cross-domain evidence. |
| Recent benchmark mis-scoring work | SHOULD CITE | **P2/OPTIONAL if posted days before deadline.** Benefit of doubt applies. |
| WebGym | OPTIONAL | **OP/P3.** Cite only if space permits. |

---

# 7. Five highest benefit-to-effort edits

1. **Delete the title TODO and `.*`** — tiny effort; removes immediate submission failure.
2. **Replace the 22% sentence with the two-condition-union/per-condition wording** — tiny effort; fixes a headline numerical error.
3. **Replace “Every mode…” with the exact 44/48 realised-success statement** — tiny effort; removes a direct contradiction with Table 7.
4. **Rename “lower bound” and replace “none improves” with comparator-specific wording** — small effort; reconciles the abstract with Table 28 and prevents a central credibility hit.
5. **Replace “cost ceiling/routing” with the retrospective one-bit oracle sentence** — small effort; preserves the strongest result while removing three predictable reviewer objections at once.

---

## Final submission judgment

After P0 and the highest-priority P1 wording fixes, the manuscript is scientifically defensible as a controlled measurement and negative-result paper. The remaining largest work item is making the six confirmed leakage corrections primary and regenerating dependent products. If that regeneration cannot be completed before upload, the paper should at minimum avoid using any leakage-sensitive red·B2 result as a headline or as evidence for a universal claim; however, knowingly retaining confirmed invalid successes as the primary outcome remains a material review risk.
