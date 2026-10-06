---
type: literature
status: abstract-level
created: 2026-10-07
purpose: prior work for the budget-routing / early-stop line (实验笔记 §505.21–§505.28) — checked before any novelty wording, as §505.27 required
---

# Budget routing / early stop for agents — precedents (checked 2026-10-07)

**Depth: abstracts only** (arXiv abstract pages, read 2026-10-07). Every "they do / do not"
below is from the abstract; before a sentence in the paper rests on a difference, read the
full text of that paper.

## Verdict

The family is **populated, and recently** (most entries are May–Sep 2026). "Stop spending on
episodes that will fail" is not ours to claim. Two of the papers even report savings in the
same range as §505.21. What the abstracts do *not* show anyone doing, and what the paper
could rest on instead:

1. **Pre-flight, no model call, assigned as a cap, not an abort.** Every early-stop entry
   below decides *online*, from the trajectory prefix or hidden states. §505.21 decides
   before step 0 from the task and its start page, and gives a short cap rather than
   killing the episode, which keeps early successes.
2. **Read against same-condition reruns.** §505.21 holds on 14/18 rerun pairs (train on run
   A, apply to run B). None of the abstracts measures whether the saving survives rerun noise.
3. **The contrast with representation routing.** The same pre-flight difficulty signal that
   fails at *which mode* (§505.18–§505.19) works for *how long*. That contrast is this
   project's, and it rests on the 6-mode × 11-cell grid.
4. **Our own online-abort result is negative** (§459: per-step learned abort ≈ null on this
   data), while *Doomed from the Start* (hidden states) and *Monitoring Web Agents*
   (observable prefixes, WebArena-Lite) report positive online prediction. Reviewers will
   see the tension, so the paper has to address it (different signal, backbone, benchmark).

## Entries

| paper | when | signal / timing | setting | action | headline (abstract) |
|---|---|---|---|---|---|
| *Doomed from the Start: Early Abort of LLM Agent Episodes via a Recall-Controlled Probe Cascade* — Ruan et al., arXiv 2607.06503 | 2026-07 | linear probes on hidden states, from the first round; online | TextCraft, WebShop | abort, recall-controlled | tokens −60.2% (TextCraft) / −54.9% (WebShop) at 90% recall |
| *BAGEN: Are LLM Agents Budget-Aware?* — Lin et al., arXiv 2606.00198 | 2026-05 | agent predicts an interval on its remaining budget at each step; prompted, then SFT+RL; online | four environments (not named in abstract) | early stop / alert | 28–64% tokens saved on failed trajectories |
| *Fail-Fast, Restart-Smart* — Wang et al., arXiv 2608.03222 | 2026-08 | observable trajectory prefix; online | SWE-bench Verified | abort + restart | 14.6–20.4% execution tokens at target FPR |
| *Monitoring Web Agents Without Internal Signals* — Pan et al., arXiv 2609.02057 | 2026-09 | observable trajectory prefix; online | WebArena-Lite, Online-Mind2Web | early intervention at fixed false-cut budget | no savings number in abstract |
| *Step-level Optimization for Efficient Computer-use Agents* — Wei et al., arXiv 2604.27151 | 2026-04 | stuck / milestone monitors; online | computer-use benchmarks | escalate small → strong model | no number in abstract |
| *Predicting Task Difficulty Without Rollouts* — Krsteski & Meyer, arXiv 2608.05797 | 2026-08 | task description and pre-rollout features; **pre-execution** | 17 agentic benchmarks | prediction only (curricula); no budget allocation stated | — |

Seen in search, not read: *EarlyEval* (arXiv 2609.02783, early outcome prediction for cheaper
evaluation), *Automata from Agent Traces* (2608.23670), *When Does Learning to Stop Help?*
(2606.30852, reasoning models), *ARES* (2603.07915, per-step reasoning-effort routing for
agents), *Budget-Aware LLM Agents* (ResearchGate 405474946).

## Already in the repo, for the model-side axis

`routing/compass_artifact_…md` and `routing/Systematizing-Efficiency-…md` cover early exit
and adaptive depth *inside the model* (layers, thinking budget). That is a different axis
from capping environment steps, and those notes do not cover the agent-level entries above.
