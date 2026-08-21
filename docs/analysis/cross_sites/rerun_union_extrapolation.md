# Six reruns of one mode vs six distinct modes

> Regenerate: `.venv/bin/python3 scripts/analysis/rerun_union_extrapolation.py`

Answers the one question the noise-floor inventory declines to answer by
arithmetic: the ceiling adds five arms, the measured floor adds one, so what
would five *rerun* arms have bought? Model and its bias direction are
documented in the script header. Reported as an upper bound on repetition.

Cell `classifieds·B0`, n=224. Best single mode 27.23%; six-mode oracle **43.30%**.

| replicated arm | single-run SR | discordant | flippable (est) | U(2) model | U(2) observed | **U(6) model** |
|---|---:|---:|---:|---:|---:|---:|
| `B0.cls.dom` (DOM) | 17.41% | 27 | 54 | 23.44% | 22.32% | **29.09%** |
| `B0.cls.vision` (Vision) | 25.00% | 32 | 64 | 32.14% | 31.70% | **38.84%** |
| `B0.cls.som` (SoM) | 27.23% | 29 | 58 | 33.71% | 34.82% | **39.77%** |
| `B0.cls.ptext` (P-text) | 15.62% | 23 | 46 | 20.76% | 20.09% | **25.57%** |
| `B0.cls.pprompt` (P-prompt) | 19.64% | 28 | 56 | 25.89% | 24.55% | **31.75%** |
| `B0.cls.psom` (P-SoM) | 15.62% | 27 | 54 | 21.65% | 20.98% | **27.30%** |

**U(2) model vs observed** differs by -1.12 to +1.34pp — the model is checked out of sample rather than assumed, and it errs on the side of crediting repetition too much.

**The comparison at matched arm count.** Six reruns of one mode reach 25.57–39.77%. Six distinct modes reach **43.30%** — a residual of only **3.53pp** over the best six-rerun account.

⇒ Two readings, and both belong in the write-up.

**At matched ARM COUNT the residual does not clear our own threshold.** 3.53pp sits below the 3.82–4.15pp one-sided band derived in §1b, so six distinct representations are not distinguishable from six repetitions of the strongest one. Repetition explains most of the ceiling. This is the strongest available attack on the ceiling claim and it substantially lands.

**At matched SERVING COST the two are not interchangeable.** The mode oracle spends ONE episode per task; the six-rerun union spends SIX. So the residual buys little, but it buys it at a sixth of the deployment cost — which is the axis a deployment actually pays on. The surviving claim is therefore about cost-efficiency of the ceiling, not about its height.

⚠️ Both readings are bounded by one cell — `classifieds·B0` is the only cell carrying replicated arms. Neither generalises without more replicates.
