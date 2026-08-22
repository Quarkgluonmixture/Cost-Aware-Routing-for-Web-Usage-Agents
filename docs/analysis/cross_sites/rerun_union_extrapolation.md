# Six reruns of one mode vs six distinct modes

> Regenerate: `.venv/bin/python3 scripts/analysis/rerun_union_extrapolation.py`

Answers the one question the noise-floor inventory declines to answer by
arithmetic: the ceiling adds five arms, the measured floor adds one, so what
would five *rerun* arms have bought? The model and the reason p=1/2 is
NOT a bound are documented in the script header. Reported as an
identified interval plus a p=1/2 sensitivity point.

Cell `classifieds·B0`, n=224. Best single mode 27.23%; six-mode oracle **43.30%**.

| replicated arm | single-run SR | discordant | flippable (est) | U(2) model | U(2) observed | **U(6) model** |
|---|---:|---:|---:|---:|---:|---:|
| `B0.cls.dom` (DOM) | 17.41% | 27 | 54 | 23.44% | 22.32% | **29.09%** |
| `B0.cls.vision` (Vision) | 25.00% | 32 | 64 | 32.14% | 31.70% | **38.84%** |
| `B0.cls.som` (SoM) | 27.23% | 29 | 58 | 33.71% | 34.82% | **39.77%** |
| `B0.cls.ptext` (P-text) | 15.62% | 23 | 46 | 20.76% | 20.09% | **25.57%** |
| `B0.cls.pprompt` (P-prompt) | 19.64% | 28 | 56 | 25.89% | 24.55% | **31.75%** |
| `B0.cls.psom` (P-SoM) | 15.62% | 27 | 54 | 21.65% | 20.98% | **27.30%** |

**U(2) model vs observed** differs by -1.12 to +1.34pp (model minus observed). This checks the union arithmetic and the run-to-run stability of the marginal. It does NOT check p: U(2) = |A| + d/2 for every p, so the residual is algebraically (|A|-|B|)/2 and carries no information about the flip model's shape.

**The comparison at matched arm count.** At p=1/2, six reruns of one mode reach 25.57–39.77%, leaving a residual of 3.53pp under the six-mode oracle (**43.30%**). But p is not identified: over the feasible range the best arm's U(6) spans 35.72–54.33%, which straddles the oracle. The point comparison is a sensitivity, not a settled result.

⇒ Two readings, and both belong in the write-up.

**At matched ARM COUNT, what survives is a share, not a residual.** Across the identified interval six reruns of the best arm recover 52.8–168.6% of the 16.07pp headroom. ⚠️ The lower end is a plug-in quantity at the observed moments, NOT a confidence bound: a paired-task bootstrap puts it above one half in only ~59% of resamples. Report it as a model-based sensitivity.

**At matched SERVING COST the two are not interchangeable.** The mode oracle spends ONE episode per task; the six-rerun union spends SIX. So the residual buys little, but it buys it at a sixth of the deployment cost — which is the axis a deployment actually pays on. The surviving claim is therefore about cost-efficiency of the ceiling, not about its height.

⚠️ Both readings are bounded by one cell — `classifieds·B0` is the only cell carrying replicated arms. Neither generalises without more replicates.
