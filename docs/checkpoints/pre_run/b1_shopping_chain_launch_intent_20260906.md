# Launch intent — B1 × VWA-shopping × {Vision, P-text, P-prompt} (2026-09-06)

Same discipline as `floor_chain_launch_intent_20260817.md`,
`reframe_chain_launch_intent_20260819.md`, `b1_reddit_chain_launch_intent_20260826.md`
and `b0_reddit_replicate_chain_launch_intent_20260902.md` (§469.7).

⚠️ **Timing, stated honestly.** The chain went out at **11:08:56 UTC**; this file was
written ~30 min later. It is therefore *not* pre-fire in the strict sense the four
earlier intents were. It **is** written before any episode exists — the first cell
needs ~57 h — so every landing point below is still fixed before a number can
influence it. The user asked for the GPU not to sit idle; that is why the order is
fire-then-declare this once, and the deviation is recorded rather than smoothed over.

**Every cell below gets reported, whatever its number says.** A cell that lands and
is omitted requires a written reason in this file, in the same commit.

## What this chain is for — a third site, not a tighter number

B1 × VWA-shopping currently has **3 of 6 arms**: `dom` 4.83 / `som` 7.59 /
`P-SoM` 4.60 (435 ep each, 2026-08-09/12/14). This chain supplies `vision`,
`P-text`, `P-prompt` and makes shopping the **third site where B1 has all six arms**
(cls and reddit already do).

⚠️ **What it explicitly does NOT buy: a noise-calibrated envelope.** The 2^6
unique-solve envelope (§470.3) needs *two* runs per arm. Shopping has no replicate on
any arm, and this chain adds none. Every reading below is therefore a **single-run**
reading, and per the rerun-band rule (CLAUDE.md ⭐) a single-run cross-side difference
cannot be read against zero. Stating this up front because "third site lands" is
exactly the phrasing that would otherwise get quietly upgraded to "third site
confirms".

### Reading 1 — does the cross-side pattern reappear on a third site?

The surviving hero is a **cross-side** coverage difference: visual side {SoM, Vision}
vs text side {P-text, P-prompt, P-SoM}. B1's two existing sites disagree about how
strong it is:

| B1 | SoM | Vision | P-text | P-prompt | P-SoM |
|---|---|---|---|---|---|
| cls | 14.29 | 12.50 | 7.59 | 6.70 | 6.70 |
| reddit | 8.29 | 2.93 | 6.83 | 6.34 | 6.83 |

On cls the visual side is clearly above the text side on **both** arms. On reddit
`Vision` **collapses to 2.93** — below every text arm — so the visual side is carried
by `SoM` alone. Shopping is the tiebreaker.

| B1·shop·Vision lands at | what it means |
|---|---|
| **≈ SoM (6-9%)** | the cls pattern (both visual arms high) reappears; reddit's Vision collapse is that site's own property. |
| **< 4% (near the text arms)** | the visual side is carried by `SoM` alone on 2 of 3 sites. The hero must then be stated as a **SoM-vs-text** difference, not a visual-side-vs-text-side one. |
| **between (4-6%)** | no site-level rule is available on B1; the honest claim is per-site and that is what gets written. |

### Reading 2 — is the text side internally flat here too?

On both existing sites the three text arms sit within ~0.9pp of each other
(cls 6.70-7.59, reddit 6.34-6.83). Shopping already has `P-SoM` = 4.60.

| P-text / P-prompt land at | what it means |
|---|---|
| within ~1pp of 4.60 | the "text side is internally flat" reading holds on all three sites — the strongest form available without replicates. |
| spread > 2pp | format/prompt-style **does** move SR on shopping, and the flatness claim becomes site-conditional. ⚠️ This is the outcome easiest to explain away as noise (no replicate exists to bound it), so it is written down first. |

### Reading 3 — what the N=2-sites caveat can and cannot become

B-1295 carries an "N=2 sites" caveat. Landing this chain makes it N=3 **on B1 only**.
B0 × shopping has `dom`/`som`/`vision` but **no phantom arms**, and adding them was
judged to buy nothing (`task_shop_expansion`: "§5 存活的 cost ceiling adds no arm").

⇒ The softened caveat is **asymmetric** and must be written that way: *B1* spans three
sites with six arms; *B0* spans two. Any sentence of the form "the pattern holds across
three sites" without naming the backbone is false.

## The cells, in fire order

| # | Cell | n | Est. wall-clock | Cost |
|---|---|---|---|---|
| **S1** | B1 × shop × `vision` | 435 | ~57 h | **$0** (local bf16) |
| **S2** | B1 × shop × `phantom_text` | 435 | ~57 h | **$0** |
| **S3** | B1 × shop × `phantom_prompt` | 435 | ~57 h | **$0** |

Wall-clock from the three landed B1×shop cells: 67.3 / 47.7 / 57.3 h ⇒ ~57 h mean,
**~7.1 days total → ~2026-09-13**, against ARR **2026-10-12** (29 days spare).

**Ordering is load-bearing.** `vision` is the visual side's only gap, so one cell
completes one whole side. If the chain is cut, the survivor is "visual side complete +
text side 2/3", which still supports Reading 1 — the primary question. Running the two
text arms first would leave the visual side permanently short and make Reading 1
unanswerable, i.e. the fallback would be a *different* test rather than a weaker one.

## Power, declared up front

`d ≈ n × SR × 0.59` is a **replicate** quantity and does **not** apply here — these are
first runs, not reruns. The relevant precision is the SR interval: at n=435 and
SR ≈ 5%, SE ≈ 1.04pp ⇒ 95% CI ≈ **±2.0pp**. Reading 2's "within ~1pp" band is therefore
**inside** one cell's own CI, and cannot be resolved by these runs alone. Reading 2 is
declared **descriptive only** before it runs.

⚠️ **Denominator to check on landing, not assume**: the three landed shop cells each
carry **435** episodes, while `_status/cells/cell_b0_shop_*.md` frontmatter says
`n: 466` and [[reference-benchmark-task-sizes]] records a scored set amended
435→432 (AM_10). Collection vs scored vs frontmatter disagree by up to 34. **Do not
pick one here** — resolve against `preregistration.md` + `canonical_task_universe.py`
when the first cell lands (this is the B-1992 confusion, one site over).

## Halt conditions

- any cell finishes with episodes ≠ **435** (the denominator the three landed shop
  cells used; see the caveat above before treating a mismatch as a failure)
- another site chain found running → refuse (host-global lease, hard rule #3)
- wall-clock past **2026-09-20** → halt regardless of progress (leaves 3 weeks to ARR)
- no quota gate: B1 runs on local weights and spends nothing on the proxy

## Known and accepted — do NOT re-diagnose mid-flight

- `.locks/manifest_bind_halt.marker` stands (2026-09-06 08:26, from the B0×reddit
  replicate chain). It gates only `experiment_watchdog.py` RESUME_MISSING and
  `queue_phase1_paper_grade.sh`; neither `queue_chain.sh` nor `queue_baseline.sh`
  reads it (verified 2026-09-02). **Consequence accepted:** the watchdog will not
  auto-refill missing episodes, so the 435-episode check is the backstop.
- WA-shopping shares the **same docker container** as VWA-shopping (7770/7780). The
  WA chain queued behind this one therefore cannot start until this releases
  `p79_magento.lock`.

## Prediction, recorded before the fact

From the cls/reddit patterns: `Vision` ≈ **5-8%** (near `som` 7.59), `P-text` and
`P-prompt` ≈ **4-5.5%** (near `P-SoM` 4.60). That prediction **supports** the hero —
the cross-side difference reappearing on a third site.

Writing it down matters because it is the self-serving one. The row that would cost
most is `Vision` **< 4%**: it would turn the hero from "visual side vs text side" into
"SoM vs text", on 2 of 3 sites. Predicting the comfortable outcome in advance is what
makes the uncomfortable one reportable if it arrives.
