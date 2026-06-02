# Cross-mode Router Features — B0 classifieds (深化 §306)

> **Source**: `scripts/analysis/cross_mode_routable_deepdive.py` over 6 canonical fire runs
> (dom R21557 / som R5313 / vision R32024 / ptext R31183 / psom R32031 / pprompt R14655),
> deterministic (no sub-agent). Raw tables → `B0_classifieds_6mode_routable_deepdive.md`.
> Deepens §306 first-look (routable 88 / oracle 43.3% / +16pp) into router-actionable signal.
>
> ⚠️ **PROVISIONAL — NOT paper-grade.** Single (B0, classifieds) run. The 6-mode oracle is a
> MAX over 6 modes → **one-sided upward-biased** by the B0 serving floor (§308 within-baseline
> paired = **13.3%**; §302 vision replicate = **14.3%**), which is the **same order of magnitude
> as the +16pp oracle lift**. Every magnitude here (oracle, lift, exclusive counts) is an
> **UPPER BOUND** and needs replicate-calibration (§293 MC perturbation of the success matrix)
> before it can be cited. Per advisor 2026-05-29 (§309) the noise is accepted as a *disclosed
> limitation* for workshop scope, but this digest still marks every fragile number. **What
> travels without a replicate = the feature→class DIRECTION (§2 table); what does NOT = the
> oracle MAGNITUDE.** See §4.

---

## 1. Section 6 narrative — classifieds routing value is **visual listing recognition**

The single durable mechanism behind the routable space on classifieds is **thumbnail / listing
recognition**, and it cleanly separates the two representation CLASSES:

- **image-modes** = `{som, vision}` — agent sees pixels (annotated image / raw screenshot)
- **text-modes** = `{dom, phantom_text, phantom_som, phantom_prompt}` — pixel-blind (AXTree / regex text; phantom = *skip annotated image*)

**Image-recognition failure spectrum** (THUMBNAIL + IMG failures from the §306 taxonomy §2
matrix; fewer = better at recognizing the correct listing):

| mode | THUMBNAIL | IMG | THUMB+IMG | class |
|---|---:|---:|---:|---|
| som | 24 | 8 | **32** | IMAGE |
| vision | 26 | 6 | **32** | IMAGE |
| phantom_som | 35 | 10 | 45 | text |
| phantom_prompt | 32 | 15 | 47 | text |
| phantom_text | 35 | 12 | 47 | text |
| dom | 37 | 16 | **53** | text |

The two image-modes (32 each) cluster **13–21 failures below** every text-mode (45–53). This is
a **2-tier separation by pixel access**, not a smooth `dom>som>vision` ordering — within the
image tier `som`(24) and `vision`(26) THUMBNAIL counts are tied inside the ~13% serving floor, so
the honest statement is **text-modes ≫ image-modes on listing recognition**, with `dom` the worst
(pixel-blind + no SoM marks). THUMBNAIL dominates the routable failures (24–37 per mode) while the
detail-page perception failures (IMG, 6–16) and pure-navigation failures (SEARCH-NAV, 3–15) are
small — the routing value lives at the **list-page thumbnail layer**, confirming §291/§306.

**Site-asymmetry — a PREDICTION, not yet a result.** Classifieds is a thumbnail-dense marketplace,
so visual features should dominate routing here (this digest). Reddit (text-forum posts, few
listing thumbnails) should show a **flatter** image-vs-text gap, and shopping (also thumbnail-rich
but different layout) an intermediate one. **We have only the cls endpoint** — the asymmetry claim
requires the reddit + shop 6-mode runs before §6 can assert it. State as: *"on classifieds the
routable axis is visual; whether visual features remain the dominant router signal on reddit/shop
is the falsifiable site-asymmetry test (pending)."*

---

## 2. Router feature candidate table (the durable deliverable)

Task-intrinsic features available to a learned router **before** the episode runs (from
`task_config`), cross-tabbed against mode-CLASS capability. "img_only" = image-class solves AND
text-class **all** fail (= must route to pixels); "txt_only" = reverse. Class gap = img-class SR −
text-class SR over all tasks with that feature value.

| # | feature = value | routing signal | class gap | img_only : txt_only | strength | mechanism |
|---|---|---|---:|---:|---|---|
| 1 | `eval_type = string_match` | → **IMAGE** | **+16pp** (42 vs 26) | 15 : 5 (3×) | **strong** | extract-answer requires first visually locating the right listing among thumbnails |
| 2 | `has_image = False` (no input photo) | → **IMAGE** | **+10pp** (29 vs 19) | **29** : 14 | **strong** | with no reference photo the agent must visually parse the page itself; largest img_only pool |
| 3 | `has_image = True` (input photo given) | → **TEXT ok** | −5pp (46 vs 51) | 4 : 7 | medium (counterintuitive) | the reference image is *textualizable* via intent → pixels add little; do NOT auto-route to vision |
| 4 | `overall_difficulty = easy` | → **IMAGE** | **+21pp** (59 vs 38) | 12 : 4 | strong-but-small-N (N=39) | on easy tasks the differentiator is "can you see the page" |
| 5 | `visual_difficulty = hard` | → IMAGE (mild) | +8pp (33 vs 25) | 13 : 6 | weak-but-**monotone** (easy+1 → med+5 → hard+8) | isolates the perception axis: harder perception → more pixel value |
| 6 | `eval_type = program_html` | → **CHEAP text** | −3pp (16 vs 19) | 2 : 3 | medium (N=31) | state/action tasks; both classes low (23/31 unsolved) → spend the cheapest mode, save image budget |

**Composite rule sketch** (for the learned router, §6): route to image-class when
`(eval_type == string_match) OR (has_image == False)` and **not** `program_html`; this covers
most of the 33 img_only tasks. The `visual_difficulty`/`overall_difficulty` interaction is the
subtle part — **route to pixels when visual difficulty is high AND overall/reasoning difficulty is
not the bottleneck**; once reasoning dominates (overall=hard) both classes collapse to ~22% and
pixels stop rescuing, so visual features lose discriminative power.

> **Feature-importance caveat**: these are *class-level* (image vs text) signals, deliberately
> coarser than per-mode routing because class-level absorbs single-mode serving flips better
> (§4). A learned router will also have cheap runtime signals (first-obs token count, AXTree size)
> not covered here; this table is the **task-intrinsic prior**, not the full feature set.

---

## 3. Oracle marginal — a 3-mode portfolio captures 92% of the ceiling

Greedy-by-SR cumulative oracle (where the +16pp actually comes from):

| order | mode | SR | marginal NEW | cumulative oracle |
|---|---|---:|---:|---:|
| 1 | som | 27.2% | +61 | 61 (27.2%) |
| 2 | vision | 25.0% | +15 | 76 (**33.9%**) |
| 3 | phantom_prompt | 19.6% | +13 | 89 (**39.7%**) |
| 4 | dom | 17.4% | +4 | 93 (41.5%) |
| 5 | phantom_text | 15.6% | +2 | 95 (42.4%) |
| 6 | phantom_som | 15.6% | +2 | 97 (43.3%) |

- The **two image-modes alone (som+vision) = 76/97 = 78%** of the oracle ceiling; adding
  `phantom_prompt` (the one text-mode with real marginal, +13) reaches **89/97 = 92%** with a
  3-mode portfolio. `dom`/`phantom_text`/`phantom_som` add only +4/+2/+2 — **near-redundant for
  the oracle** (their value is cost, not coverage).
- Router-design implication for §6: a **small heterogeneous portfolio** (image pair + one
  divergent text-mode) is almost sufficient; the learned router's job is mostly **image-vs-text
  class selection + one phantom tie-breaker**, not a full 6-way decision. This also shrinks the
  noise surface (fewer modes → fewer one-sided flips inflating the union).

---

## 4. Noise sensitivity — what survives the ~13–14% serving floor

| quantity | value | noise exposure | citable now? |
|---|---:|---|---|
| best-single SR (som) | 27.2% | LOW — single mode, ~symmetric flips → ≈unbiased | yes (with floor disclosure) |
| 6-mode oracle SR | 43.3% | HIGH — MAX is one-sided, picks up every positive flip | **only as upper bound** |
| oracle lift | +16.1pp | HIGH — inherits oracle's one-sided bias | **only as upper bound** |
| exclusive-solves (k=1) | 29 | **HIGHEST** — one fail→pass flip fabricates a spurious exclusive (§302 decomposed 14/224 ≈ 6% fail→pass per replicate, concentrated near-boundary) | **no — replicate first** |
| shared-solves (k≥2) | 59 | MEDIUM — needs ≥2 modes to agree | partially |
| img_only / txt_only | 33 / 21 | MEDIUM — class-level (any-of-2 / any-of-4) absorbs single-mode flips | direction yes, count no |
| feature→class direction (§2) | — | LOW — systematic representation effect, survives symmetric scatter | **yes — the durable output** |

**Why the +16pp is an upper bound, precisely.** A single mode's SR is roughly unbiased: serving
noise flips some true-solves to fail and some true-fails to solve, ~symmetrically. But the oracle
is `MAX` over 6 modes — it **keeps every spurious fail→pass flip and discards every spurious
pass→fail flip**. So on the 127 near/universal-fail tasks, any one of 6 modes catching a lucky
flip inflates the union. The exclusive count (29) is the worst case: it is *defined* as "only one
mode solved", which is exactly the shape a lone noise flip produces. The §306/§293 prescription
stands: **replicate ≥2 clean runs per mode → MC-perturb the success matrix at the measured flip
rate → report the oracle-lift distribution and P(lift > floor)**, before any of §3's magnitudes
enter the paper. Under §309 (workshop scope) this is a disclosed limitation, not a blocker — but
§6 prose must phrase the lift as *"single-run oracle ceiling, reproducibility-caveated"*, never as
a measured routing gain.

---

## 5. Pointers

- Raw deepening tables (reproducible): `B0_classifieds_6mode_routable_deepdive.md`
- First-look it deepens: 笔记 §306 + `B0_classifieds_6mode_failure_taxonomy.md`
- Framework + 3-mode origin: 笔记 §291 + `scripts/analysis/cross_mode_failure_taxonomy.py`
- Noise floor provenance: §302 (vision 14.3% / codex cold-start) + §308 (paired 13.3%, B1=0)
- Statistical prescription: §293 (replicate-calibrated MC) · advisor scope §309
- Feeds: paper §6 (router substrate — feature table + portfolio finding) · §3 site-asymmetry test
- Constraint: cross-mode quantitative analysis under discover-then-freeze; all numbers PROVISIONAL
  until the 6-mode freeze + reddit/shop replicates land.
