# Cross-Site Pattern Consolidation (Classifieds vs Reddit)

This note consolidates the existing `disagreement_clusters.md` B0 and B1 compact diagnostics by site. It does not re-run any trajectories. Task audit labels come from `docs/analysis/cross_sites/codex_audit_classifieds.json` and `docs/analysis/cross_sites/codex_audit_reddit.json`: A = text-only, B = reference-image visual, C = page-screenshot visual, D = uncertain.

## Overview

Source coverage:

| Model | Site | Disagreement tasks | Failure-side pairs | Phantom included |
|---|---:|---:|---:|---|
| B0 | classifieds | 36 | 108 | yes, but Phantom-SoM steps are trace-unavailable |
| B0 | reddit | 18 | 54 | yes, but Phantom-SoM steps are trace-unavailable |
| B1 | classifieds | 33 | 66 | no Phantom runs |
| B1 | reddit | 12 | 24 | no Phantom runs |

Main consolidation:

- Most mechanism-level clusters are cross-site, not site-only: visual-missing, early-finish/wrong-commit, SoM visual-hijack/click-loop, Vision click-loop/no-text-grounding, element-misground, abandon-after-N, and search-loop each appear in both classifieds and reddit with at least three task instances after pooling B0+B1.
- The strongest capability shift, SoM visual-hijack/click-loop, is cross-site but site-modulated: B1 increases it by **+50.0 pp** on classifieds and **+33.3 pp** on reddit.
- DOM visual-missing is the most stable invariant: B0->B1 rises from 71.0% to 76.2% on classifieds and from 80.0% to 100.0% on reddit in the DOM failure slice.
- The one strict site-only cluster in the current taxonomy is B1 classifieds `click-loop` in DOM, a narrow OSClass listing/action-loop case. Larger site differences are better described as prevalence shifts rather than new mechanisms.

## Site-Agnostic Patterns

Criteria: the pattern appears in both classifieds and reddit with at least three unique `(model, task)` instances per site. `B0-64(B)` means B0 task 64 with Codex audit category B.

| Pattern | Scope | Classifieds task IDs + audit category | Reddit task IDs + audit category |
|---|---|---|---|
| visual-missing | DOM-only | B0-11(D), B0-49(C), B0-52(B), B0-60(B), B0-61(B), B0-64(B), B0-93(B), B0-101(B), B0-106(C), B0-112(C), B0-120(C), B0-130(C), B0-132(C), B0-149(C), B0-152(C), B0-160(C), B0-165(B), B0-166(B), B0-187(C), B0-192(C), B0-194(C), B0-201(D), B1-17(D), B1-44(B), B1-48(B), B1-79(B), B1-93(B), B1-110(C), B1-112(C), B1-130(C), B1-131(C), B1-135(B), B1-151(C), B1-170(B), B1-173(C), B1-174(C), B1-184(B), B1-220(D) | B0-2(B), B0-4(B), B0-7(B), B0-14(B), B0-100(C), B0-131(B), B0-139(B), B0-142(C), B0-148(C), B0-150(C), B0-152(C), B0-179(B), B1-77(C), B1-120(C), B1-131(B), B1-171(B), B1-201(B) |
| early-finish/wrong-commit | cross-mode | B0-11(D), B0-14(C), B0-16(C), B0-40(D), B0-60(B), B0-93(B), B0-98(A), B0-111(C), B0-112(C), B0-120(C), B0-149(C), B0-160(C), B0-174(C), B0-192(C), B0-201(D), B0-209(A), B0-210(A), B0-217(A), B0-222(C), B1-10(D), B1-15(C), B1-50(C), B1-79(B), B1-83(C), B1-174(C), B1-221(A) | B0-69(C), B0-79(C), B0-81(C), B0-94(C), B0-142(C), B0-148(C), B1-6(B), B1-58(C), B1-120(C), B1-189(B) |
| visual-hijack/click-loop | SoM-only | B0-64(B), B0-152(C), B0-184(B), B0-194(C), B1-10(D), B1-25(D), B1-44(B), B1-45(B), B1-64(B), B1-101(B), B1-110(C), B1-151(C), B1-164(B), B1-189(C), B1-196(D), B1-210(A), B1-220(D) | B0-7(B), B0-79(C), B0-100(C), B0-150(C), B1-0(B), B1-18(B), B1-100(C), B1-188(B), B1-189(B), B1-201(B) |
| click-loop/no-text-grounding | Vision-only | B0-52(B), B0-60(B), B0-101(B), B0-165(B), B0-166(B), B0-184(B), B0-210(A), B1-40(D), B1-45(B), B1-48(B), B1-64(B), B1-101(B), B1-112(C), B1-127(C), B1-170(B), B1-184(B), B1-210(A) | B0-2(B), B0-4(B), B0-124(C), B0-131(B), B1-18(B) |
| element-misground | cross-mode, Vision-dominant | B0-49(C), B0-93(B), B0-98(A), B0-106(C), B0-115(B), B0-127(C), B0-130(C), B0-132(C), B0-174(C), B0-187(C), B1-83(C), B1-111(C), B1-130(C), B1-173(C) | B0-7(B), B0-81(C), B1-77(C), B1-100(C), B1-131(B), B1-171(B), B1-188(B) |
| abandon-after-N | cross-mode | B0-16(C), B0-127(C), B0-167(B), B0-201(D), B1-17(D), B1-19(D), B1-25(D), B1-93(B), B1-131(C), B1-135(B), B1-164(B), B1-189(C) | B0-100(C), B0-139(B), B0-179(B) |
| search-loop | cross-mode | B0-61(B), B0-64(B), B0-167(B), B0-217(A), B1-40(D), B1-111(C) | B0-14(B), B0-124(C), B0-162(A) |

### Minimal Cross-Site Snippets

| Pattern | Classifieds snippet | Reddit snippet |
|---|---|---|
| visual-missing | `B0 classifieds task 11 DOM`: P5,P6,P14; 12 steps; actions scroll->scroll->scroll->scroll->scroll->click->scroll->type; reason=fail_no_progress. | `B0 reddit task 2 DOM`: P6; 3 steps; actions type->click->finish; reason=fail_finish_wrong_url_not_found. |
| early-finish/wrong-commit | `B0 classifieds task 11 SoM`: no P-rule; 1 step; actions finish; reason=fail_early_finish. | `B0 reddit task 69 SoM`: no P-rule; 4 steps; actions scroll->type->click->finish; reason=fail_finish_eval_mismatch. |
| visual-hijack/click-loop | `B0 classifieds task 64 SoM`: P14; 30 steps; actions type->type->scroll->scroll->scroll->click->scroll->scroll; reason=fail_max_steps_target_unreachable. | `B0 reddit task 7 SoM`: P14; 11 steps; actions click->scroll->scroll->click->back->click->click->click; reason=raw_success_adjusted_false. |
| click-loop/no-text-grounding | `B0 classifieds task 52 Vision`: P14; 5 steps; actions type->click->click->click->click; reason=fail_incomplete_or_stuck. | `B0 reddit task 2 Vision`: P14; 5 steps; actions click->click->click->click->click; reason=fail_incomplete_or_stuck. |
| element-misground | `B0 classifieds task 49 Vision`: P1; 6 steps; actions click->scroll->scroll->click->click->finish; reason=fail_finish_eval_mismatch. | `B0 reddit task 7 Vision`: P1,P5,P14; 8 steps; actions type->click->click->scroll->click->click->click->click; reason=raw_success_adjusted_false. |
| abandon-after-N | `B0 classifieds task 16 DOM`: P5,P14; 11 steps; actions type->scroll->scroll->scroll->click->scroll->scroll->scroll; reason=fail_no_progress. | `B0 reddit task 100 Vision`: P14; 6 steps; actions type->scroll->scroll->click->scroll->finish; reason=fail_finish_empty_answer. |
| search-loop | `B0 classifieds task 61 SoM`: P13,P14; 6 steps; actions type->scroll->type->type->type->type; reason=fail_no_progress. | `B0 reddit task 14 Vision`: P13; 12 steps; actions type->type->type->type->type->type->type->type; reason=fail_incomplete_or_stuck. |

## Site-Specific and Site-Modulated Patterns

Strict site-only clusters are rare. The major paper mechanism should therefore be framed as cross-site invariant with site-specific prevalence and trajectory expression.

| Pattern | Site | Task IDs + audit category | Environmental interpretation |
|---|---|---|---|
| DOM click-loop | classifieds only | B1-127(C) | This is a narrow OSClass listing-loop: DOM can repeatedly operate on search/listing affordances without resolving the visual target. It is not a major aggregate mechanism, but it shows how OSClass listing density can turn DOM exploration into local repeated actions. |
| trace-unavailable | B0 both sites, artifact | classifieds: B0-11(D), B0-14(C), B0-16(C), ...; reddit: B0-2(B), B0-4(B), B0-14(B), ... | This is not a site mechanism. It is a data availability artifact from cleared Phantom-SoM step traces and should be excluded from universal/site-specific mechanism claims. |
| Vision click-loop/no-text-grounding | classifieds-heavy | classifieds 17 task instances vs reddit 5 | OSClass pages have dense product cards, repeated thumbnails, and contact/listing controls. Without DOM/mark text, Vision often clicks plausible repeated cards or contact-like targets. Reddit still shows the pattern, but fewer tasks expose the same listing-grid density. |
| element-misground | reddit B1-shifted | classifieds 14 task instances vs reddit 7 overall; B1 reddit share rises to 50.0% | Postmill trajectories are slower and page/comment targets are visually similar; B1 Vision often clicks plausible but wrong post/comment regions. On classifieds the same failure exists, but B1 shifts away from element-misground toward Vision click-loop/no-text-grounding. |
| search-loop | sparse but cross-site | classifieds: B0-61(B), B0-64(B), B0-167(B), B0-217(A), B1-40(D), B1-111(C); reddit: B0-14(B), B0-124(C), B0-162(A) | Search-loop is not purely a Postmill artifact. Reddit's slower Postmill navigation can amplify repeated search, but OSClass also induces repeated search/result scanning when item listings are visually dense or the target is described by reference-image attributes. |

Audit files for lookup:

- Classifieds categories: `docs/analysis/cross_sites/codex_audit_classifieds.json`
- Reddit categories: `docs/analysis/cross_sites/codex_audit_reddit.json`

## B0->B1 Shift: Site-Disaggregated Tables

The aggregate contrast table in `disagreement_clusters.md` remains correct, but it hides site modulation. Denominators below are mode-specific failure-side pairs within each site/model slice.

### Classifieds

| Mode / pattern | B0 count | B0 share | B1 count | B1 share | Shift |
|---|---:|---:|---:|---:|---:|
| DOM visual-missing | 22/31 | 71.0% | 16/21 | 76.2% | +5.2 pp |
| DOM search-loop | 1/31 | 3.2% | 2/21 | 9.5% | +6.3 pp |
| SoM early-finish/wrong-commit | 12/18 | 66.7% | 4/18 | 22.2% | -44.4 pp |
| SoM visual-hijack/click-loop | 4/18 | 22.2% | 13/18 | 72.2% | +50.0 pp |
| Vision text/grounding loops | 7/27 | 25.9% | 10/27 | 37.0% | +11.1 pp |
| Vision element-misground | 10/27 | 37.0% | 4/27 | 14.8% | -22.2 pp |

### Reddit

| Mode / pattern | B0 count | B0 share | B1 count | B1 share | Shift |
|---|---:|---:|---:|---:|---:|
| DOM visual-missing | 12/15 | 80.0% | 5/5 | 100.0% | +20.0 pp |
| DOM search-loop | 2/15 | 13.3% | 0/5 | 0.0% | -13.3 pp |
| SoM early-finish/wrong-commit | 4/12 | 33.3% | 3/9 | 33.3% | +0.0 pp |
| SoM visual-hijack/click-loop | 4/12 | 33.3% | 6/9 | 66.7% | +33.3 pp |
| Vision text/grounding loops | 4/14 | 28.6% | 1/10 | 10.0% | -18.6 pp |
| Vision element-misground | 2/14 | 14.3% | 5/10 | 50.0% | +35.7 pp |

### Stability Interpretation

1. **Universal: DOM visual-missing.** Both sites and both capabilities show DOM failures dominated by missing visual evidence. This supports a paper claim that text-only DOM cannot cover visual-bound tasks even when model capability changes.
2. **Cross-site but site-modulated: SoM visual-hijack flip.** The aggregate +43.7 pp B1 shift is not classifieds-only: classifieds contributes +50.0 pp and reddit contributes +33.3 pp. The mechanism is cross-site, but OSClass listing density makes the shift stronger.
3. **Site-modulated: Vision failure subtype.** B1 classifieds shifts Vision toward no-text-grounding loops (+11.1 pp), while B1 reddit shifts toward element-misground (+35.7 pp). This suggests Vision's weakness is universal, but its surface form depends on site UI: OSClass repeated listings versus Postmill post/comment target localization.
4. **Non-universal: DOM search-loop in the exclusive slice.** DOM search-loop is sparse and changes direction by site (+6.3 pp on classifieds, -13.3 pp on reddit). Section 4 should avoid claiming that DOM search-loop dominates disagreement failures universally; it is better used as a whole-run trajectory-gradient finding, with disagreement failures skewing toward visual-missing.

## Implications for Section 4/5 Claims

- Safe universal claim: representation determines failure geometry across sites: DOM loses visual evidence, SoM can be hijacked by marked visual affordances, and Vision lacks stable text grounding.
- Safe capability claim: the B0->B1 SoM visual-hijack flip is cross-site, not a classifieds-only artifact, but its magnitude is larger on OSClass.
- Site-modulated claim: the same representation weakness manifests through different UI surfaces: OSClass listing density amplifies listing/card loops; Postmill's slower forum/comment navigation and visually similar post targets amplify localization errors.
- Avoid overclaim: search-loop is not a universal disagreement-task signature. It remains useful for the broader strategy-gradient figure and reddit macro behavior, but within one-arm-only disagreement tasks it is lower-frequency than visual-missing and click/grounding failures.
