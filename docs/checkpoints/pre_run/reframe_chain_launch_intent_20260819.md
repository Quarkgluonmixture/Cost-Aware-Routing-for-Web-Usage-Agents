# Launch intent — reframe chain (declared 2026-08-19, BEFORE fire)

Same discipline as `floor_chain_launch_intent_20260817.md` and for the same reason
(codex Mode B F2 / §469.7): **cell 2 below is a same-condition replicate**, and a
replicate only counts as one by being listed in `CLEAN_PAIRS`. Registering after the
number is known makes registration a choice. So the list is fixed here first.

**Every cell below gets registered / reported, whatever its number says.** A cell that
lands and is omitted requires a written reason in this file, in the same commit.

## What this chain is for

Two claims carry the reframe, and each has exactly one soft spot. This chain buys
evidence at those two spots and nowhere else.

| Claim | Where it stands today | The soft spot | Cell that hits it |
|---|---|---|---|
| **A.** Routing gains concentrate at the image boundary; within the text side, format and prompt change *which* tasks succeed, not *how many*, and the change sits inside the noise | 8 cells of SR + a noise envelope on **one** cell (`cls·B0`, §470.3) | the envelope exists for one cell only | **B** (reddit phantom floors) |
| **B.** API-served models carry an unreported reproducibility floor (B0 10.3-14.3% vs local dense 0.00%) | six B0 arms + two B1 arms, all `cls` | **one API model.** B0 is MoE *and* proxy-served; B1 is dense *and* local. The two candidate causes are perfectly confounded | **A2** (a second API model's replicate pair) |

Cell **C** serves the third thing: every backbone in the study is open-weight, so
"is the baseline strong enough" (NAACL attack surface #3) currently has no answer.

## The cells, in fire order

| # | Cell | Site | Cost | Est. | Purpose |
|---|---|---|---|---|---|
| **0** | B5 × cls × dom — **1-task smoke** | cls | ~$1 | 30 min | Gate. The response_format road is verified only on single calls (§471.5) and against a fake observation. Nothing has yet run it through the 30-step agent loop. |
| **A1** | B5 × cls × dom (full) | cls | ~$32 | ~9 h | First API model that is not B0. |
| **A2** | B5 × cls × dom — **REPLICATE of A1** | cls | ~$32 | ~9 h | **The pair.** Breaks the MoE↔serving confound: if B5 also lands near 12%, the floor is a property of API serving; if near 0%, it is specific to B0/Qwen-MoE and every effect size measured on B0 needs its own discount. |
| **B1-B3** | B0 × red × {P-text, P-prompt, P-SoM} | **red** | ~$66 | **5.5-8 d** | Second site for the noise envelope. `cls` phantom floors are what let §470.3 bound the unique-solve counts; reddit has none. |
| **C1-C5** | B5 × cls × {som, vision, P-text, P-prompt, P-SoM} | cls | ~$160 | ~2.2 d | Attack surface #3 + whether the phantom space reproduces across model families. **Trimmable** — dropping modes weakens this but breaks nothing else. |

**Total ~$291.** Budget is not the binding constraint (>$600 available); wall-clock is.

## Power, declared up front

`d ≈ n × SR × 0.59` (§468 / B-1972); `d < 10` ⇒ inventory, not an interval.

- **A2** (the pair that matters): B5's SR on `cls·dom` is **unknown** — this is the
  first run of the model. If it lands near B0's dom SR (17.4%) then d ≈ 23 and the
  pair carries an interval. **If B5's SR comes in below ~8%, d < 10 and the pair is
  descriptive only.** Declared now so the outcome cannot be reinterpreted later.
- **B1-B3** (reddit phantom): d ≈ 13.0-15.9 at the archived reddit phantom SRs — above
  the bar, which is why these three and not the B1 reddit arms (d ≈ 3.0-8.9).

## What is deliberately NOT in this chain

- **B0 × red × {dom, som, vision}** — the reddit *phantom* arms are the gap; the
  AXTree/image arms there can wait.
- **sol / luna tiers of GPT-5.6.** One tier only (`terra`), so "B5" names one model.
  luna is 10x cheaper and would do for A1/A2 (which need only success/steps), but
  mixing tiers across cells would make B5 two models and nothing cross-cell comparable.
  If terra's SR lands clearly above B0's, attack surface #3 is answered and `sol` is
  unnecessary; `sol` is only needed if terra ≈ B0 and the question becomes "was the
  tier strong enough".
- **Anything on shopping or WA.** Direction-dependent; waits for the 09-07 verdict.

## Known risks, stated before the fact

1. **B5 has never run a real episode.** Cell 0 exists to catch that, and the chain
   halts there if it fails. The specific unknowns: whether the model emits `finish`,
   whether it loops, whether `element_id`s read off a real AXTree dispatch, whether
   the JSON survives a 7K-char observation.
2. **B5 has no logprobs at all** (`logprobs_unavailable: true`, §471.5). Its confidence
   column is verbalized-only. **Any cross-baseline confidence comparison must exclude
   B5 or restrict itself to `verbalized`.** This is a disclosure, not a defect —
   declared in the config and stamped on every step record.
3. **reddit is the schedule risk.** 4.6 ep/h measured (44 h/cell), and the archive holds
   two reddit phantom cells at **154 h and 184 h**. Three cells could take 8 days rather
   than 5.5. It is placed third rather than last so that an overrun still leaves room.
4. **The verdict date has flipped twice** (09-07 → 08-21 → 09-07, §470.9). This chain is
   planned against **09-07**. If it flips again the tail (C) is what gets cut.

## Halt conditions (the chain stops itself; no one is watching it)

- cell 0 smoke fails its gate → halt before any paid cell
- any cell finishes with episodes ≠ expected (224 cls / 203 red) → halt, do not proceed
- cumulative recorded cost exceeds **$400** → halt (planned $291; the ceiling catches a
  pricing surprise like the one §471.2 found, where a recorded price was wrong by 6x)
- wall-clock passes **2026-09-06** → halt regardless of progress
- another site chain is found running → refuse to launch (host-global lease)
