#!/usr/bin/env python3
"""Offline prior-work-style router baselines for the full-OOF B0 cell.

This analysis is deliberately isolated from the preregistered router and H10
artifacts.  It reuses the locked B0_classifieds task folds, fold-local TF-IDF /
MI-18 feature assets, canonical Pass-1 outcomes, and
``total_billed_cost_usd``.  The RouteLLM-, FrugalGPT-, Vardanyan-, and
LazyMCoT-style labels denote offline adaptations, not faithful reproductions of
the source methods.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import StandardScaler


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from p79.experiment.io_utils import read_jsonl_dedup  # noqa: E402
from p79.policies.learned_router import (  # noqa: E402
    build_runtime_feature_vector,
    load_selected_idx_fold,
    load_vectorizer_fold,
)
from p79.policies.router_features import (  # noqa: E402
    MODES as ORACLE_MODE_ORDER,
    derive_oracle_label,
)
from scripts.analysis.router_offline_replay import (  # noqa: E402
    DISPLAY_MODES,
    build_offline_raw_features,
    collect_cell_outcomes,
    fold_map_sha256,
    load_paper_grade_entries,
    policy_metrics,
    sha256_file,
)
from scripts.analysis.router_pareto_analysis import (  # noqa: E402
    PolicyPoint,
    dominance_relation,
    pareto_frontier,
)


CELL_ID = "B0_classifieds"
N_FOLDS = 5
K_VALUES = (5, 10, 20)
RANDOM_REPETITIONS = 1_000
RANDOM_SEED = 20_260_715
TAU_QUANTILES = tuple(round(float(value), 2) for value in np.linspace(0.0, 1.0, 21))

SOURCE_DIR = REPO / "results/phantom_paper/l1_router_offline_20260715"
DEFAULT_REPLAY = SOURCE_DIR / "router_offline_replay.json"
DEFAULT_PARETO = SOURCE_DIR / "pareto/router_pareto_analysis.json"
DEFAULT_SUCCESS_OOF = SOURCE_DIR / "pareto/cost_aware_success_oof.json"
DEFAULT_OUT_DIR = SOURCE_DIR / "prior_baselines"
DEFAULT_REPORT = (
    REPO
    / "docs/checkpoints/codex_outputs/router_litmined_baselines_2026-07-15.md"
)
PHASE1_ROOT = REPO / "results/visualwebarena/phase1"
FORBIDDEN_CANONICAL_OUT = (REPO / "results/phantom_paper/l1_router").resolve()

SCHEMA_VERSION = "2026-07-15-router-litmined-baselines-offline-v2"
ARTIFACT_STATUS = "OFFLINE / NON-GATE / POST-HOC EXPLORATORY"
DISCLAIMER = (
    "OFFLINE / NON-GATE / POST-HOC EXPLORATORY — prior-work-style adaptations "
    "replaying Pass-1 trajectories; not the preregistered live H10 gate"
)

CONFIDENCE_AGGREGATIONS = {
    "mean_logprob": ("mean_logprob", "mean"),
    "min_logprob": ("min_logprob", "min"),
    "mean_margin": ("mean_margin", "mean"),
    "min_margin": ("min_margin", "min"),
}
CONFIDENCE_PRIORITY = tuple(CONFIDENCE_AGGREGATIONS)

LAZYMCOT_START_MODE = "vision"
LAZYMCOT_ESCALATION_MODE = "som"
LAZYMCOT_DIFFICULTY_QUANTILE = 0.50
VARDANYAN_START_MODE = "dom"
VARDANYAN_ESCALATION_MODE = "vision"

# Local-source-only mining inventory.  Rows intentionally preserve UNVERIFIED
# flags from the notes instead of upgrading them from model memory.
LITERATURE_MINING_ROWS = (
    ("RouteLLM", "query → weak/strong model", "能（已实现）：fold-local TF-IDF/MI-18 cosine kNN 对 six-mode oracle label 投票。", "docs/checkpoints/deliverables/chapter_literature_review.md:62-64; docs/checkpoints/paper_drafts/paper.bib:154-163"),
    ("FrugalGPT", "query → cheap-to-expensive model cascade", "能（已实现）：按 fold-train cost 排 mode，用观测 confidence 升级并累计全部轨迹 cost。", "docs/checkpoints/deliverables/chapter_literature_review.md:62-64; docs/checkpoints/paper_drafts/paper.bib:166-175"),
    ("Hybrid LLM", "query → small/large model", "不能：routed object 不同；six-mode 没有 edge/cloud model arms。", "docs/checkpoints/paper_drafts/paper.bib:963-971"),
    ("CSCR / Cost-Aware Contrastive Routing", "query → dynamic LLM pool", "不能：routed object 不同，且需 model fingerprints/contrastive encoder；现有 kNN 仅作 RouteLLM-style 近似。", "docs/checkpoints/paper_drafts/paper.bib:732-740"),
    ("WebRouter", "query → model（ca-VIB）", "不能：本地 verified bib 与 scan 对 routed object 冲突，且无 ca-VIB objective/training recipe；UNVERIFIED 方法细节。", "docs/checkpoints/paper_drafts/paper.bib:974-981; docs/literature/routing/literature_scan_output.md:5-22"),
    ("Adaptive VLM Routing", "GUI action → VLM pool", "不能：routed object 是 model capability，且需 probe-model confidence/safety/live GUI。", "docs/checkpoints/deliverables/chapter_literature_review.md:62-64; docs/checkpoints/paper_drafts/paper.bib:503-507"),
    ("Runaway is Ashamed / agent early exit", "episode → halt/continue", "不能：需要 live completion verification/step-budget takeover，不是离线选 observation arm。", "docs/checkpoints/deliverables/chapter_literature_review.md:62-64; docs/checkpoints/paper_drafts/paper.bib:776-784"),
    ("Dynamic Model Routing and Cascading survey", "六范式均为 query → model", "不能：survey 不是 producer；difficulty/preference/clustering/RL/UQ/cascade 六类均选 model。", "docs/checkpoints/paper_drafts/paper.bib:1031-1038"),
    ("DMR", "agent step → bf16/quantized precision", "不能：没有同任务的 precision-arm trajectories，且需 KL labels + RL。", "docs/checkpoints/deliverables/chapter_literature_review.md:66-68; docs/checkpoints/paper_drafts/paper.bib:820-828"),
    ("PANDO", "agent compute → planner/actor capacity + skills", "不能：routed object/agent stack 不同，skills、compression、cache 与 routing 不可从 six-mode replay 分离。", "docs/checkpoints/deliverables/chapter_literature_review.md:66-68; docs/checkpoints/paper_drafts/paper.bib:798-806"),
    ("ReVision", "screenshot history → retained visual patches", "不能：需 patch masks 与 fine-tuned backbone；Pass-1 没有 filtered-visual arm。", "docs/checkpoints/deliverables/chapter_literature_review.md:66-68; docs/checkpoints/paper_drafts/paper.bib:809-817"),
    ("Avenir-Web", "grounding request → grounding expert", "不能：routed object 是 expert，replay 没有 per-expert outputs。", "docs/checkpoints/deliverables/chapter_literature_review.md:66-68; docs/checkpoints/paper_drafts/paper.bib:189-197"),
    ("ModServe", "multimodal request/stage → serving resource", "不能：serving scheduling/resource disaggregation，不产生 task-level observation choice。", "docs/checkpoints/deliverables/chapter_literature_review.md:66-68; docs/checkpoints/paper_drafts/paper.bib:200-208"),
    ("Read More, Think More", "model capability/budget → HTML vs A11y guideline", "能但不新增：B0-Classifieds 单 model/预算下退化为一个 fixed policy，与已有 fixed 点重合。", "docs/checkpoints/paper_drafts/aaai27/aaai27_main.md:93-93; docs/checkpoints/paper_drafts/paper.bib:1041-1048"),
    ("LazyMCoT / Focus When Necessary", "natural-image sample → direct vs localized re-observation", "能（新实现）：fold-train task-length 中位数作难度 proxy，Vision→SoM；累计两段 cost。", "docs/checkpoints/paper_drafts/paper.bib:1071-1078; docs/literature/raw/2026-06-21-digest.md:31-35"),
    ("Vardanyan browser report", "A11y-first → selective vision", "能（新实现）：DOM step 失败代理触发 Vision，累计两段独立轨迹 cost。", "docs/checkpoints/paper_drafts/paper.bib:1131-1138; docs/literature/raw/2026-05-29-digest.md:59-65"),
    ("Agent-E", "task → DOM-text variant", "不能：six-mode 不含 text_only/input_fields/all_fields 三种 DOM 变体。", "docs/checkpoints/paper_drafts/paper.bib:380-385"),
    ("BoundaryRouter", "query → direct LLM/full agent via early memory", "不能：需 cold-start dual-system experience memory且强调 no-gold routing；replay 只有 gold task outcomes。", "docs/checkpoints/paper_drafts/paper.bib:1081-1088; docs/literature/raw/2026-06-05-digest.md:20-29"),
    ("Adaptive Re-Ranking", "query → BM25/MiniLM/BGE tier", "能但不新增：oracle-vs-learned-router 形态已由 six-mode oracle + locked LR/cost-aware OOF curve 覆盖；IR routed object 不同。", "docs/checkpoints/paper_drafts/paper.bib:1091-1098; docs/literature/raw/2026-06-28-digest.md:24-24"),
    ("FocusAgent", "AXTree → retained relevant lines", "不能：Pass-1 没有 retriever-pruned AXTree trajectories。", "docs/checkpoints/deliverables/chapter_literature_review.md:100-104; docs/checkpoints/paper_drafts/paper.bib:178-186"),
    ("Revisiting Observation Reduction / MFS", "HTML → reduced HTML variant", "不能：需 11 个 reduction outputs/MFS coverage；six-mode 无对应 arms。", "docs/checkpoints/paper_drafts/paper.bib:1061-1068"),
    ("Sema", "screen stream → structured text + compact visual tokens", "不能：replay 没有 Sema compact-token hybrid arm。", "docs/checkpoints/paper_drafts/paper.bib:1121-1128"),
    ("MirageDetect", "image-question → answer/abstain/escalate", "不能：需 image + extra CLIP forward/layer features，offline task-text replay 不具备。", "docs/checkpoints/paper_drafts/paper.bib:1111-1118"),
    ("Signal-Driven Observation", "event signal → when to reread DOM", "不能：position paper、无实现；routes WHEN rather than WHICH representation，且需 live DOM events。", "docs/checkpoints/paper_drafts/paper.bib:1101-1108"),
    ("Plan-Then-Execute", "task → program/runtime LLM subroutine", "不能：typed-API execution architecture 不在 six-mode trajectory menu。", "docs/checkpoints/paper_drafts/paper.bib:1051-1058"),
    ("WebChallenger / PageMem", "page → selected structured-DOM sections", "不能：需 offline exploration/PageMem/workflows；是 intra-representation attention。", "docs/checkpoints/paper_drafts/paper.bib:510-517"),
    ("VisualWebArena observation ablations", "run config → AXTree/screenshot/SoM/HTML", "能但不新增：静态配置已经就是 six fixed policies，不是 runtime router。", "docs/literature/routing/compass_artifact_wf-a2769e2a-a0f4-463f-8c0c-17d2916bb323_text_markdown.md:71-74"),
    ("SeeAct", "deployment choice → grounding view", "能但不新增：静态 grounding comparison 由 fixed DOM/SoM/Vision 点近似；无 per-task selector。", "docs/literature/routing/compass_artifact_wf-a2769e2a-a0f4-463f-8c0c-17d2916bb323_text_markdown.md:73-75"),
    ("WebVoyager", "deployment choice → screenshot+SoM vs text-only", "能但不新增：全程 fixed modality，对应 fixed-point comparison。", "docs/literature/routing/compass_artifact_wf-a2769e2a-a0f4-463f-8c0c-17d2916bb323_text_markdown.md:74-76"),
    ("D2Snap / Beyond Pixels", "DOM → downsampled DOM", "不能：无 downsampled-DOM trajectories；不是多-mode selector。", "docs/literature/routing/The recent literature clusters into eight routing axes; the first six are fairly well populated, while true per-task input-represe.md:14-19"),
    ("AgentOccam", "observation/history → aligned trimmed text", "不能：无其 rewritten/history-selected observation arm。", "docs/literature/routing/deep-research-report.md:19-22"),
    ("A11y-Compressor", "A11y nodes → compact A11y", "不能：无 compressed-A11y trajectories。", "docs/literature/routing/Systematizing-Efficiency-in-Multimodal-Agents-From-Model-Cascades-to-Input-Repre.md:63-73"),
    ("LineRetriever", "DOM/AXTree → navigation-relevant lines", "不能：无 retriever outputs/trajectories。", "docs/literature/routing/The recent literature clusters into eight routing axes; the first six are fairly well populated, while true per-task input-represe.md:10-19"),
    ("UIFormer", "DOM/tree → DSL-transformed UI representation", "不能：无 synthesized transform outputs；UNVERIFIED scan-only detail。", "docs/literature/routing/literature_scan_output.md:5-21"),
    ("Less is More / SimpAgent", "screenshot/history → compressed visual stream", "不能：within-visual compression，replay 无该 arm。", "docs/literature/routing/compass_artifact_wf-a2769e2a-a0f4-463f-8c0c-17d2916bb323_text_markdown.md:77-89"),
    ("ShowUI", "screenshot → retained UI visual tokens", "不能：需 model-internal token selection；不是 six-mode task selector。", "docs/literature/routing/compass_artifact_wf-a2769e2a-a0f4-463f-8c0c-17d2916bb323_text_markdown.md:38-41; docs/literature/routing/compass_artifact_wf-a2769e2a-a0f4-463f-8c0c-17d2916bb323_text_markdown.md:80-81"),
    ("GUI-KV", "GUI tokens → retain/evict KV entries", "不能：缺 KV traces/cache policy arms。", "docs/literature/routing/Systematizing-Efficiency-in-Multimodal-Agents-From-Model-Cascades-to-Input-Repre.md:52-61"),
    ("Skip-Vision", "visual tokens → bypass FFN/KV", "不能：model-internal training/inference path 不可由 episode replay重建。", "docs/literature/routing/Systematizing-Efficiency-in-Multimodal-Agents-From-Model-Cascades-to-Input-Repre.md:52-61"),
    ("PruneVid", "video/visual tokens → retained tokens", "不能：within-representation token pruning；无 web six-mode arm。", "docs/literature/routing/compass_artifact_wf-a2769e2a-a0f4-463f-8c0c-17d2916bb323_text_markdown.md:38-41"),
    ("SCOPE token pruning", "visual tokens → saliency/coverage subset", "不能：within-visual pruning；需 token saliency。", "docs/literature/routing/compass_artifact_wf-a2769e2a-a0f4-463f-8c0c-17d2916bb323_text_markdown.md:38-41"),
    ("GEAR", "KV cache → compressed cache", "不能：cache compression 不产生 selectable observation trajectory。", "docs/literature/routing/deep-research-report.md:17-20"),
    ("Fit and Prune", "visual tokens → retained subset", "不能：缺 token-level states；routed object 不同。", "docs/literature/routing/deep-research-report.md:17-20"),
    ("Spatio-Temporal Token Pruning", "GUI tokens → retained spatio-temporal subset", "不能：缺 token masks/filtered trajectories。", "docs/literature/routing/deep-research-report.md:17-20"),
    ("iSHIFT", "screen → fast/global vs slow/local perception", "能但未新增：概念上可做 Vision→SoM 两段升级，已由 LazyMCoT-style 点覆盖；原法需 perception tokens/model支持。", "docs/literature/routing/Systematizing-Efficiency-in-Multimodal-Agents-From-Model-Cascades-to-Input-Repre.md:111-124"),
    ("RecAgent", "GUI → filtered components/human fallback", "不能：需 component recommender/human-in-loop；UNVERIFIED 方法细节。", "docs/literature/routing/Systematizing-Efficiency-in-Multimodal-Agents-From-Model-Cascades-to-Input-Repre.md:111-124"),
    ("DiMo-GUI", "GUI image → text/icon paths + adaptive halt", "不能：需 live iterative zoom/separate visual paths。", "docs/literature/routing/Systematizing-Efficiency-in-Multimodal-Agents-From-Model-Cascades-to-Input-Repre.md:33-41; docs/literature/routing/Systematizing-Efficiency-in-Multimodal-Agents-From-Model-Cascades-to-Input-Repre.md:111-124"),
    ("V-GEMS / See and Remember", "page → DOM-text agent vs vision agent", "不能：dual-agent binary switch，且论文/id 在本地明确 UNVERIFIED。", "docs/literature/routing/compass_artifact_wf-a2769e2a-a0f4-463f-8c0c-17d2916bb323_text_markdown.md:82-90"),
    ("AMuFC", "claim → text-only vs multimodal verifier", "不能：fact-check domain/routed pipeline different；无 web trajectories。", "docs/literature/routing/Systematizing-Efficiency-in-Multimodal-Agents-From-Model-Cascades-to-Input-Repre.md:75-84"),
    ("VistaGUI / MP-GUI", "screen → graphic/text/spatial perception paths", "不能：需 modality extractors/adaptive fusion gate。", "docs/literature/routing/Systematizing-Efficiency-in-Multimodal-Agents-From-Model-Cascades-to-Input-Repre.md:75-84"),
    ("Quantifying and Mitigating Modality Preference", "input → modality weights", "不能：routes internal weighting，非 discrete observation arm；UNVERIFIED transfer。", "docs/literature/routing/Systematizing-Efficiency-in-Multimodal-Agents-From-Model-Cascades-to-Input-Repre.md:75-84"),
    ("Camel Browser Toolkit", "task demand → SoM/DOM/text interaction", "不能：本地仅概述、无可复现 policy/论文细节，标 UNVERIFIED。", "docs/literature/routing/Systematizing-Efficiency-in-Multimodal-Agents-From-Model-Cascades-to-Input-Repre.md:105-108"),
    ("AdaptAgent", "demonstration → text-only vs multimodal", "不能：routes adaptation demonstrations rather than task observation；本地无完整方法细节。", "docs/literature/routing/Systematizing-Efficiency-in-Multimodal-Agents-From-Model-Cascades-to-Input-Repre.md:105-109"),
    ("Accio", "web task → speculative profile fast path/full ReAct", "不能：需 live profile matching + verification + environment handoff。", "docs/literature/routing/Systematizing-Efficiency-in-Multimodal-Agents-From-Model-Cascades-to-Input-Repre.md:43-50"),
    ("LLM Cascades with Mixture of Thoughts", "reasoning answer → accept/escalate model", "不能：需多 thought samples/answer consistency；routed object 不同。", "docs/literature/routing/Systematizing-Efficiency-in-Multimodal-Agents-From-Model-Cascades-to-Input-Repre.md:21-31"),
    ("Cascadia", "requests → serving cascade plan/resources", "不能：system load/resource allocation 不在 replay。", "docs/literature/routing/Systematizing-Efficiency-in-Multimodal-Agents-From-Model-Cascades-to-Input-Repre.md:21-31"),
    ("Unified Routing and Cascading", "query → model/cascade", "不能：cross-model routed object；本地仅 taxonomy。", "docs/literature/routing/The recent literature clusters into eight routing axes; the first six are fairly well populated, while true per-task input-represe.md:5-8"),
    ("Route-To-Reason", "query → model + reasoning strategy", "不能：six-mode replay 没有 model/strategy factorial arms。", "docs/literature/routing/The recent literature clusters into eight routing axes; the first six are fairly well populated, while true per-task input-represe.md:5-8"),
    ("EMAFusion", "query → LLM selection/integration", "不能：routed object 是 models；本地仅 taxonomy。", "docs/literature/routing/The recent literature clusters into eight routing axes; the first six are fairly well populated, while true per-task input-represe.md:5-8"),
    ("Adaptive Reasoning Executor", "query → reasoning executor/strategy", "不能：无相应 compute arms；本地仅 taxonomy。", "docs/literature/routing/The recent literature clusters into eight routing axes; the first six are fairly well populated, while true per-task input-represe.md:5-8"),
    ("Is Escalation Worth It?", "query → pairwise cascade deferral", "不能：UNVERIFIED local entry，且 cross-model pairwise envelope。", "docs/literature/routing/compass_artifact_wf-a2769e2a-a0f4-463f-8c0c-17d2916bb323_text_markdown.md:23-26"),
    ("CALM", "token/example → compute depth", "不能：model layer halting，不可从 episode trajectories replay。", "docs/literature/routing/The recent literature clusters into eight routing axes; the first six are fairly well populated, while true per-task input-represe.md:7-8"),
    ("EESD / Speculative Decoding via Early-exiting", "token → exit/draft verification", "不能：decoding-internal routed object。", "docs/literature/routing/The recent literature clusters into eight routing axes; the first six are fairly well populated, while true per-task input-represe.md:7-8"),
    ("Dr.LLM", "token → dynamic layer route", "不能：model-internal depth，缺 layer states。", "docs/literature/routing/The recent literature clusters into eight routing axes; the first six are fairly well populated, while true per-task input-represe.md:7-9"),
    ("AdaPonderLM", "token/example → ponder depth", "不能：adaptive compute depth，非 observation selector。", "docs/literature/routing/The recent literature clusters into eight routing axes; the first six are fairly well populated, while true per-task input-represe.md:7-9"),
    ("ADEPT", "token → early-exit depth", "不能：UNVERIFIED future-dated method，缺 layer-level traces。", "docs/literature/routing/compass_artifact_wf-a2769e2a-a0f4-463f-8c0c-17d2916bb323_text_markdown.md:28-31"),
    ("MuE / You Need Multiple Exiting", "VLM input → encoder/decoder exits", "不能：requires early-exit model architecture。", "docs/literature/routing/compass_artifact_wf-a2769e2a-a0f4-463f-8c0c-17d2916bb323_text_markdown.md:28-31"),
    ("SpecRouter", "draft tokens → speculative route", "不能：token-level draft/verify path 不在 replay。", "docs/literature/routing/The recent literature clusters into eight routing axes; the first six are fairly well populated, while true per-task input-represe.md:8-9"),
    ("Fast Speculative Edge-Cloud Decoding", "tokens → edge draft/cloud verify/exit", "不能：serving + token-level pipeline 不同。", "docs/literature/routing/The recent literature clusters into eight routing axes; the first six are fairly well populated, while true per-task input-represe.md:8-9"),
    ("Medusa", "tokens → parallel draft heads/base verification", "不能：UNVERIFIED model-internal decoding method。", "docs/literature/routing/compass_artifact_wf-a2769e2a-a0f4-463f-8c0c-17d2916bb323_text_markdown.md:33-36"),
    ("SDSAT", "tokens → semantic adaptive drafts", "不能：decoding-token object，非 task observation。", "docs/literature/routing/compass_artifact_wf-a2769e2a-a0f4-463f-8c0c-17d2916bb323_text_markdown.md:33-36"),
    ("LLMLingua", "prompt → retained high-information tokens", "不能：需要 compressor outputs；无对应 trajectories。", "docs/literature/routing/Systematizing-Efficiency-in-Multimodal-Agents-From-Model-Cascades-to-Input-Repre.md:63-73"),
    ("LongLLMLingua", "long context → retained key information", "不能：within-prompt compression，缺 compressed arms。", "docs/literature/routing/compass_artifact_wf-a2769e2a-a0f4-463f-8c0c-17d2916bb323_text_markdown.md:43-46"),
    ("LLMLingua-2", "prompt token → keep/drop", "不能：requires token classifier/compressed trajectories。", "docs/literature/routing/compass_artifact_wf-a2769e2a-a0f4-463f-8c0c-17d2916bb323_text_markdown.md:43-46"),
    ("FastMMoE", "tokens/experts → retained path", "不能：model/expert internals，且本地仅 taxonomy。", "docs/literature/routing/The recent literature clusters into eight routing axes; the first six are fairly well populated, while true per-task input-represe.md:9-10"),
    ("KVCrush", "KV entries → retain/drop", "不能：缺 cache entries/states；本地仅 taxonomy。", "docs/literature/routing/The recent literature clusters into eight routing axes; the first six are fairly well populated, while true per-task input-represe.md:9-10"),
    ("KVReviver", "KV entries → retain/revive", "不能：cache-level routed object。", "docs/literature/routing/The recent literature clusters into eight routing axes; the first six are fairly well populated, while true per-task input-represe.md:9-10"),
    ("RAP", "tokens/cache → retained subset", "不能：token/cache route，缺 internal states。", "docs/literature/routing/The recent literature clusters into eight routing axes; the first six are fairly well populated, while true per-task input-represe.md:9-10"),
    ("Active Context Compression", "history/context → retained context", "不能：无 compressed-context trajectories；本地仅 taxonomy。", "docs/literature/routing/The recent literature clusters into eight routing axes; the first six are fairly well populated, while true per-task input-represe.md:10-12"),
    ("BEST-Route", "query → model + sample count", "不能：routed object 是 model/test-time compute，非 representation。", "docs/literature/routing/deep-research-report.md:23-25"),
    ("OptiRoute / Dynamic Tool Routing", "task → tool/model", "不能：工具池/model池与 six-mode arms 不同；本地无可复现细节。", "docs/literature/routing/compass_artifact_wf-a2769e2a-a0f4-463f-8c0c-17d2916bb323_text_markdown.md:60-63"),
    ("Router-R1", "multi-round query → model pool/aggregation", "不能：UNVERIFIED，需 RL 与多模型调用。", "docs/literature/routing/compass_artifact_wf-a2769e2a-a0f4-463f-8c0c-17d2916bb323_text_markdown.md:60-63"),
    ("MTRouter", "multi-turn history → sequential model", "不能：UNVERIFIED scan-only；routed object 是 model。", "docs/literature/routing/literature_scan_output.md:5-13"),
    ("Surfer-H", "workflow → specialized policy/localizer/validator", "不能：UNVERIFIED scan-only，需不同 modules/models。", "docs/literature/routing/literature_scan_output.md:5-13"),
    ("Topaz", "workflow → skill/budget allocation", "不能：UNVERIFIED scan-only；无 skill profiles。", "docs/literature/routing/literature_scan_output.md:5-13"),
    ("CASTER", "workflow subtask → multi-agent route", "不能：UNVERIFIED scan-only；routed object 是 subagent。", "docs/literature/routing/literature_scan_output.md:5-13"),
    ("ECCOS", "request → heterogeneous model pool/schedule", "不能：UNVERIFIED scan-only；serving/model axis。", "docs/literature/routing/literature_scan_output.md:5-13"),
    ("OmniParser", "deployment → screenshot-only parser", "能但不新增：固定 visual design，不是 per-task router；对应 Vision-side fixed comparison。", "docs/literature/routing/The recent literature clusters into eight routing axes; the first six are fairly well populated, while true per-task input-represe.md:11-21"),
    ("WebSight", "deployment → screenshot-only web agent", "不能：fixed-modality system，不输出 per-task switch。", "docs/literature/routing/The recent literature clusters into eight routing axes; the first six are fairly well populated, while true per-task input-represe.md:11-21"),
    ("MolmoWeb", "deployment → screenshot-only web agent", "不能：fixed-modality design；本地 arXiv id 未 surfaced，UNVERIFIED。", "docs/literature/routing/The recent literature clusters into eight routing axes; the first six are fairly well populated, while true per-task input-represe.md:11-21"),
    ("AndroidWorld M3A/SeeAct", "run config → A11y vs SoM", "能但不新增：static mobile-GUI comparison，对应 fixed points；非 web/runtime router。", "docs/literature/routing/deep-research-report.md:31-35"),
    ("OSWorld-Human", "run config → SS/SS+A11y/SS+A11y+SoM", "能但不新增：static desktop comparison，不是 selector。", "docs/literature/routing/deep-research-report.md:31-36"),
    ("Set-of-Mark Prompting", "deployment → raw screenshot vs marked screenshot", "能但不新增：定义一个 representation；已有 SoM/Vision fixed points。", "docs/literature/routing/Systematizing-Efficiency-in-Multimodal-Agents-From-Model-Cascades-to-Input-Repre.md:116-125"),
)


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO))
    except ValueError:
        return str(path.resolve())


def _point_from_metric(
    policy_id: str,
    label: str,
    category: str,
    metric: dict[str, Any],
) -> dict[str, Any]:
    n_tasks = int(metric["n_tasks"])
    n_success = int(metric["n_success"])
    sum_cost = float(metric["sum_total_billed_cost_usd"])
    if n_tasks <= 0 or not 0 <= n_success <= n_tasks:
        raise ValueError(f"Invalid counts for {policy_id}: {n_success}/{n_tasks}")
    if not math.isfinite(sum_cost) or sum_cost < 0:
        raise ValueError(f"Invalid cost for {policy_id}: {sum_cost!r}")
    return asdict(
        PolicyPoint(
            policy_id=policy_id,
            label=label,
            category=category,
            mean_cost_usd=sum_cost / n_tasks,
            success_rate_pct=100.0 * n_success / n_tasks,
            n_tasks=n_tasks,
            n_success=n_success,
        )
    )


def performance_gap_recovered(
    success_rate_pct: float,
    best_single_success_rate_pct: float,
    oracle_success_rate_pct: float,
) -> float | None:
    """Offline RouteLLM-style PGR adapted to best-single and oracle anchors."""

    denominator = oracle_success_rate_pct - best_single_success_rate_pct
    if abs(denominator) <= 1e-12:
        return None
    return (success_rate_pct - best_single_success_rate_pct) / denominator


def efficiency_metrics(
    success_rate_pct: float,
    mean_cost_usd: float,
    *,
    best_single_success_rate_pct: float,
    best_single_mean_cost_usd: float,
    oracle_success_rate_pct: float,
) -> dict[str, float | None]:
    delta_sr = success_rate_pct - best_single_success_rate_pct
    delta_cost = mean_cost_usd - best_single_mean_cost_usd
    return {
        "delta_vs_best_single_sr_pp": delta_sr,
        "delta_vs_best_single_cost_usd": delta_cost,
        "performance_gap_recovered": performance_gap_recovered(
            success_rate_pct,
            best_single_success_rate_pct,
            oracle_success_rate_pct,
        ),
        "delta_sr_pp_per_delta_usd": (
            None if abs(delta_cost) <= 1e-12 else delta_sr / delta_cost
        ),
    }


def _attach_efficiency(
    point: dict[str, Any],
    *,
    best_sr: float,
    best_cost: float,
    oracle_sr: float,
) -> dict[str, Any]:
    return {
        **point,
        **efficiency_metrics(
            float(point["success_rate_pct"]),
            float(point["mean_cost_usd"]),
            best_single_success_rate_pct=best_sr,
            best_single_mean_cost_usd=best_cost,
            oracle_success_rate_pct=oracle_sr,
        ),
    }


def aggregate_episode_confidence(
    steps: Iterable[dict[str, Any]],
) -> dict[str, float | None]:
    """Mirror the existing episode aggregation for four B0 confidence fields."""

    result: dict[str, float | None] = {}
    rows = list(steps)
    for signal, (source_key, operation) in CONFIDENCE_AGGREGATIONS.items():
        values: list[float] = []
        for step in rows:
            confidence = step.get("confidence")
            if not isinstance(confidence, dict) or confidence.get(source_key) is None:
                continue
            value = float(confidence[source_key])
            if math.isfinite(value):
                values.append(value)
        if not values:
            result[signal] = None
        elif operation == "mean":
            result[signal] = float(np.mean(values))
        elif operation == "min":
            result[signal] = float(np.min(values))
        else:  # pragma: no cover - constant contract above
            raise AssertionError(operation)
    return result


def _canonical_steps_path(summary_path: Path) -> Path:
    suffix = "_summary_v2.json"
    if not summary_path.name.endswith(suffix):
        raise ValueError(f"Unexpected episode summary path: {summary_path}")
    return summary_path.with_name(
        summary_path.name.removesuffix(suffix) + "_steps_v2.jsonl"
    )


def trajectory_failure_signals(steps: Iterable[dict[str, Any]]) -> dict[str, bool]:
    """Extract observable, outcome-free escalation signals from one trajectory.

    The decision deliberately excludes evaluator reward and episode ``success``.
    Every signal is available from the step stream at or before the point at which
    the cheap trajectory stops.
    """

    parse_failure = False
    action_failure = False
    invalid_agent_action = False
    two_step_no_change = False
    empty_finish = False
    unchanged_streak = 0
    for step in steps:
        parse_failure |= step.get("parse_valid") is False
        action_failure |= step.get("action_success") is False
        invalid_agent_action |= step.get("valid_agent_action") is False
        if step.get("page_changed") is False:
            unchanged_streak += 1
            two_step_no_change |= unchanged_streak >= 2
        else:
            unchanged_streak = 0

        if step.get("action_type") == "finish":
            action = step.get("action")
            if isinstance(action, dict):
                answer_keys = [
                    key for key in ("answer", "text", "final_answer") if key in action
                ]
                if answer_keys and all(
                    action[key] is None or not str(action[key]).strip()
                    for key in answer_keys
                ):
                    empty_finish = True
    return {
        "parse_failure": parse_failure,
        "action_failure": action_failure,
        "invalid_agent_action": invalid_agent_action,
        "two_step_no_change": two_step_no_change,
        "empty_finish": empty_finish,
    }


def vardanyan_should_escalate(steps: Iterable[dict[str, Any]]) -> bool:
    """A11y/DOM-first -> vision escalation on any logged failure proxy."""

    return any(trajectory_failure_signals(steps).values())


def difficulty_trigger(intent_token_count: int, threshold: float) -> bool:
    """Strict threshold rule used by the LazyMCoT-style difficulty proxy."""

    value = int(intent_token_count)
    if value < 0 or not math.isfinite(float(threshold)):
        raise ValueError("Difficulty trigger requires a non-negative count and finite threshold")
    return value > float(threshold)


def simulate_two_stage_task(
    task_outcomes: dict[str, dict[str, Any]],
    *,
    start_mode: str,
    escalation_mode: str,
    escalate: bool,
) -> dict[str, Any]:
    """Replay one trajectory or two independent-reset trajectories with summed cost."""

    if start_mode == escalation_mode:
        raise ValueError("Two-stage modes must be distinct")
    for mode in (start_mode, escalation_mode):
        if mode not in task_outcomes:
            raise ValueError(f"Missing two-stage outcome for mode {mode}")
    executed = [start_mode, escalation_mode] if escalate else [start_mode]
    final_mode = executed[-1]
    return {
        "final_mode": final_mode,
        "executed_modes": executed,
        "n_executed": len(executed),
        "success": bool(task_outcomes[final_mode]["success"]),
        "total_billed_cost_usd": sum(
            float(task_outcomes[mode]["cost_usd"]) for mode in executed
        ),
    }


def load_observed_confidence(
    outcomes: dict[int, dict[str, dict[str, Any]]],
) -> tuple[dict[int, dict[str, dict[str, float | None]]], dict[str, Any]]:
    """Load canonical step JSONL and aggregate task/mode trajectory confidence."""

    values: dict[int, dict[str, dict[str, float | None]]] = {}
    coverage_counts = {signal: 0 for signal in CONFIDENCE_PRIORITY}
    mode_coverage = {
        mode: {signal: 0 for signal in CONFIDENCE_PRIORITY} for mode in DISPLAY_MODES
    }
    total_steps = 0
    confidence_steps = 0
    for task_id in sorted(outcomes):
        values[task_id] = {}
        for mode in DISPLAY_MODES:
            summary_path = REPO / outcomes[task_id][mode]["summary_path"]
            steps_path = _canonical_steps_path(summary_path)
            steps = read_jsonl_dedup(
                steps_path,
                summary_path=summary_path,
                strict_identity=True,
            )
            if not steps:
                raise ValueError(f"No canonical steps for {CELL_ID}/{mode}/task {task_id}")
            aggregated = aggregate_episode_confidence(steps)
            values[task_id][mode] = aggregated
            total_steps += len(steps)
            confidence_steps += sum(isinstance(step.get("confidence"), dict) for step in steps)
            for signal, value in aggregated.items():
                if value is not None:
                    coverage_counts[signal] += 1
                    mode_coverage[mode][signal] += 1

    n_episodes = len(outcomes) * len(DISPLAY_MODES)
    usable = [
        signal for signal in CONFIDENCE_PRIORITY if coverage_counts[signal] == n_episodes
    ]
    probe = {
        "summary_level_fields_present": False,
        "source": "canonical *_steps_v2.jsonl step_record.confidence",
        "aggregation": {
            "mean_logprob": "mean of step mean_logprob",
            "min_logprob": "minimum of step min_logprob",
            "mean_margin": "mean of step mean_margin",
            "min_margin": "minimum of step min_margin",
        },
        "n_task_mode_episodes": n_episodes,
        "n_steps": total_steps,
        "n_steps_with_confidence_dict": confidence_steps,
        "step_confidence_coverage": confidence_steps / total_steps,
        "episode_coverage_count_by_signal": coverage_counts,
        "episode_coverage_fraction_by_signal": {
            signal: count / n_episodes for signal, count in coverage_counts.items()
        },
        "mode_coverage_count_by_signal": mode_coverage,
        "complete_observed_signals": usable,
        "primary_signal": usable[0] if usable else None,
        "fallback_required": not bool(usable),
        "interpretation": (
            "Episode summaries do not embed confidence aggregates; the canonical "
            "step schema does. Values are aggregated with the existing confidence "
            "analysis semantics, so this remains an observed post-trajectory signal."
        ),
    }
    return values, probe


def majority_oracle_vote(
    labels: Iterable[str],
    distances: Iterable[float],
    *,
    mode_order: Iterable[str] = ORACLE_MODE_ORDER,
) -> str:
    """Uniform kNN majority vote with deterministic distance/order tie breaks."""

    label_rows = list(labels)
    distance_rows = [float(value) for value in distances]
    if not label_rows or len(label_rows) != len(distance_rows):
        raise ValueError("labels/distances must be non-empty and aligned")
    counts = Counter(label_rows)
    best_count = max(counts.values())
    candidates = [label for label, count in counts.items() if count == best_count]
    if len(candidates) == 1:
        return candidates[0]
    mean_distance = {
        label: float(
            np.mean(
                [
                    distance
                    for candidate, distance in zip(label_rows, distance_rows)
                    if candidate == label
                ]
            )
        )
        for label in candidates
    }
    order = {mode: idx for idx, mode in enumerate(mode_order)}
    return min(candidates, key=lambda label: (mean_distance[label], order[label]))


def run_knn_baselines(
    outcomes: dict[int, dict[str, dict[str, Any]]],
    entries: dict[str, dict[str, Any]],
    fold_map: dict[int, int],
    source_dir: Path,
) -> dict[str, Any]:
    raw_by_task = {
        task_id: build_offline_raw_features(entries, PHASE1_ROOT, "classifieds", task_id)[0]
        for task_id in sorted(outcomes)
    }
    labels_by_task = {
        task_id: derive_oracle_label(
            {mode: bool(outcomes[task_id][mode]["success"]) for mode in DISPLAY_MODES}
        )
        for task_id in outcomes
    }
    selected = {k: {} for k in K_VALUES}
    task_records = {k: [] for k in K_VALUES}
    fold_records: dict[str, Any] = {}

    for fold_k in range(N_FOLDS):
        train_ids = sorted(
            task_id
            for task_id in outcomes
            if fold_map[task_id] != fold_k and labels_by_task[task_id] is not None
        )
        holdout_ids = sorted(task_id for task_id in outcomes if fold_map[task_id] == fold_k)
        vectorizer = load_vectorizer_fold(source_dir, fold_k)
        selected_mask, _ = load_selected_idx_fold(source_dir, fold_k)
        if vectorizer is None or selected_mask is None:
            raise FileNotFoundError(f"Missing locked MI-18 assets for fold {fold_k}")
        if int(selected_mask.sum()) != 18:
            raise ValueError(f"Fold {fold_k} selector is not MI-18")
        if len(train_ids) < max(K_VALUES):
            raise ValueError(f"Fold {fold_k} has only {len(train_ids)} labeled neighbors")

        X_train = np.vstack(
            [
                build_runtime_feature_vector(raw_by_task[task_id], vectorizer, selected_mask)
                for task_id in train_ids
            ]
        )
        X_holdout = np.vstack(
            [
                build_runtime_feature_vector(raw_by_task[task_id], vectorizer, selected_mask)
                for task_id in holdout_ids
            ]
        )
        scaler = StandardScaler().fit(X_train)
        distances = cosine_distances(scaler.transform(X_holdout), scaler.transform(X_train))
        if not np.isfinite(distances).all():
            raise ValueError(f"Non-finite cosine distance in fold {fold_k}")
        train_labels = [str(labels_by_task[task_id]) for task_id in train_ids]

        for row_idx, task_id in enumerate(holdout_ids):
            neighbor_order = np.argsort(distances[row_idx], kind="mergesort")
            for k in K_VALUES:
                indices = neighbor_order[:k]
                neighbor_labels = [train_labels[idx] for idx in indices]
                neighbor_distances = distances[row_idx, indices].tolist()
                mode = majority_oracle_vote(neighbor_labels, neighbor_distances)
                selected[k][task_id] = mode
                outcome = outcomes[task_id][mode]
                task_records[k].append(
                    {
                        "task_id": task_id,
                        "fold_k": fold_k,
                        "selected_mode": mode,
                        "success": bool(outcome["success"]),
                        "total_billed_cost_usd": float(outcome["cost_usd"]),
                        "neighbor_vote_counts": dict(sorted(Counter(neighbor_labels).items())),
                        "nearest_distance": float(neighbor_distances[0]),
                    }
                )
        fold_records[str(fold_k)] = {
            "n_train_tasks_total": sum(fold_map[task_id] != fold_k for task_id in outcomes),
            "n_train_tasks_with_oracle_label": len(train_ids),
            "n_train_tasks_without_any_success_excluded": sum(
                fold_map[task_id] != fold_k and labels_by_task[task_id] is None
                for task_id in outcomes
            ),
            "n_holdout_tasks": len(holdout_ids),
            "train_label_distribution": dict(sorted(Counter(train_labels).items())),
        }

    rows = []
    for k in K_VALUES:
        if set(selected[k]) != set(outcomes):
            raise AssertionError(f"kNN k={k} OOF coverage mismatch")
        metric = policy_metrics(outcomes, selected[k])
        point = _point_from_metric(
            f"knn_k{k}",
            f"RouteLLM-style kNN (k={k})",
            "prior_knn",
            metric,
        )
        rows.append(
            {
                **point,
                "k": k,
                "selected_mode_counts": metric["selected_mode_counts"],
                "selections_by_task": {
                    str(task_id): mode for task_id, mode in sorted(selected[k].items())
                },
                "task_records": sorted(task_records[k], key=lambda row: row["task_id"]),
            }
        )
    return {
        "protocol": {
            "outer_folds": N_FOLDS,
            "k_values": list(K_VALUES),
            "feature_space": "locked fold-local TF-IDF + MI-18 selector",
            "fold_local_scaling": "StandardScaler fit on labeled fold-train tasks",
            "distance": "cosine distance after fold-local scaling",
            "vote": "uniform majority oracle-label vote",
            "tie_break": "lower mean neighbor distance, then locked oracle prior-cost order",
            "training_rows": (
                "fold-train tasks with at least one successful Pass-1 mode; "
                "no-success tasks have no oracle label and are excluded from neighbor memory"
            ),
        },
        "fold_records": fold_records,
        "points": rows,
    }


def simulate_cascade_task(
    task_outcomes: dict[str, dict[str, Any]],
    confidence_by_mode: dict[str, float | None],
    chain: Iterable[str],
    threshold: float,
) -> dict[str, Any]:
    """Execute an offline trajectory cascade and sum every visited trajectory."""

    modes = list(chain)
    if set(modes) != set(DISPLAY_MODES) or len(modes) != len(DISPLAY_MODES):
        raise ValueError("Cascade chain must contain each mode exactly once")
    total_cost = 0.0
    executed: list[str] = []
    final_mode = modes[-1]
    for index, mode in enumerate(modes):
        executed.append(mode)
        total_cost += float(task_outcomes[mode]["cost_usd"])
        final_mode = mode
        if index == len(modes) - 1:
            break
        confidence = confidence_by_mode.get(mode)
        if confidence is None or not math.isfinite(float(confidence)):
            raise ValueError(f"Missing cascade confidence for mode {mode}")
        if float(confidence) >= threshold:
            break
    return {
        "final_mode": final_mode,
        "executed_modes": executed,
        "n_executed": len(executed),
        "success": bool(task_outcomes[final_mode]["success"]),
        "total_billed_cost_usd": total_cost,
    }


def _fold_train_quantile_threshold(values: Iterable[float], quantile: float) -> float:
    rows = np.asarray(list(values), dtype=float)
    if rows.size == 0 or not np.isfinite(rows).all():
        raise ValueError("Threshold training values must be finite and non-empty")
    if quantile == 0.0:
        return float(np.nextafter(rows.min(), -np.inf))
    if quantile == 1.0:
        return float(np.nextafter(rows.max(), np.inf))
    return float(np.quantile(rows, quantile))


def run_observed_cascades(
    outcomes: dict[int, dict[str, dict[str, Any]]],
    confidence: dict[int, dict[str, dict[str, float | None]]],
    complete_signals: Iterable[str],
    fold_map: dict[int, int],
) -> dict[str, Any]:
    signals = list(complete_signals)
    order = {mode: idx for idx, mode in enumerate(DISPLAY_MODES)}
    fold_context: dict[int, dict[str, Any]] = {}
    for fold_k in range(N_FOLDS):
        train_ids = sorted(task_id for task_id in outcomes if fold_map[task_id] != fold_k)
        mean_costs = {
            mode: float(np.mean([outcomes[task_id][mode]["cost_usd"] for task_id in train_ids]))
            for mode in DISPLAY_MODES
        }
        chain = sorted(DISPLAY_MODES, key=lambda mode: (mean_costs[mode], order[mode]))
        thresholds = {}
        for signal in signals:
            train_values = [
                float(confidence[task_id][mode][signal])
                for task_id in train_ids
                for mode in DISPLAY_MODES
            ]
            thresholds[signal] = {
                f"{quantile:.2f}": _fold_train_quantile_threshold(train_values, quantile)
                for quantile in TAU_QUANTILES
            }
        fold_context[fold_k] = {
            "n_train": len(train_ids),
            "mean_train_total_billed_cost_usd_by_mode": mean_costs,
            "ascending_cost_chain": chain,
            "threshold_by_signal_and_quantile": thresholds,
        }

    curves: dict[str, list[dict[str, Any]]] = {}
    curve_frontiers: dict[str, list[str]] = {}
    for signal in signals:
        curve: list[dict[str, Any]] = []
        for quantile in TAU_QUANTILES:
            n_success = 0
            total_cost = 0.0
            total_executed = 0
            final_counts: Counter[str] = Counter()
            task_records: list[dict[str, Any]] = []
            for task_id in sorted(outcomes):
                fold_k = fold_map[task_id]
                context = fold_context[fold_k]
                threshold = float(
                    context["threshold_by_signal_and_quantile"][signal][f"{quantile:.2f}"]
                )
                row = simulate_cascade_task(
                    outcomes[task_id],
                    {
                        mode: confidence[task_id][mode][signal] for mode in DISPLAY_MODES
                    },
                    context["ascending_cost_chain"],
                    threshold,
                )
                n_success += int(row["success"])
                total_cost += float(row["total_billed_cost_usd"])
                total_executed += int(row["n_executed"])
                final_counts[row["final_mode"]] += 1
                task_records.append(
                    {
                        "task_id": task_id,
                        "fold_k": fold_k,
                        "threshold": threshold,
                        **row,
                    }
                )
            metric = {
                "n_tasks": len(outcomes),
                "n_success": n_success,
                "sum_total_billed_cost_usd": total_cost,
            }
            point = _point_from_metric(
                f"cascade_{signal}_q{quantile:.2f}",
                f"FrugalGPT-style cascade {signal} q={quantile:.2f}",
                "prior_cascade",
                metric,
            )
            curve.append(
                {
                    **point,
                    "signal": signal,
                    "tau_quantile": quantile,
                    "threshold_semantics": "fold-train pooled task×mode confidence quantile",
                    "mean_trajectories_executed": total_executed / len(outcomes),
                    "escalated_task_fraction": sum(
                        int(row["n_executed"] > 1) for row in task_records
                    )
                    / len(outcomes),
                    "final_mode_counts": dict(sorted(final_counts.items())),
                    "task_records": task_records,
                }
            )
        curves[signal] = curve
        curve_frontiers[signal] = [
            point.policy_id
            for point in pareto_frontier(
                [
                    PolicyPoint(
                        **{key: row[key] for key in PolicyPoint.__dataclass_fields__}
                    )
                    for row in curve
                ]
            )
        ]
    return {
        "implementation_kind": "observed_post_trajectory_confidence_cascade",
        "is_true_observed_signal_cascade": True,
        "signals": signals,
        "primary_signal": signals[0],
        "tau_quantiles": list(TAU_QUANTILES),
        "fold_context": {str(key): value for key, value in fold_context.items()},
        "curves": curves,
        "curve_frontiers": curve_frontiers,
        "cost_accounting": "sum total_billed_cost_usd for every executed Pass-1 trajectory",
        "success_accounting": "success of final executed mode only",
    }


def build_preexecution_fallback(success_oof: dict[str, Any]) -> dict[str, Any]:
    """Documented no-confidence fallback; not reached when B0 step signals exist."""

    curve = []
    for row in success_oof["curve"]:
        curve.append(
            {
                **row,
                "policy_id": row["policy_id"].replace("cost_aware", "cascade_fallback"),
                "label": row["label"].replace("Cost-aware", "Pre-execution fallback"),
                "category": "prior_cascade_fallback",
            }
        )
    return {
        "implementation_kind": "preexecution_oof_success_gate_fallback",
        "is_true_observed_signal_cascade": False,
        "signals": ["oof_probability_success"],
        "primary_signal": "oof_probability_success",
        "curves": {"oof_probability_success": curve},
        "curve_frontiers": {
            "oof_probability_success": [
                policy_id.replace("cost_aware", "cascade_fallback")
                for policy_id in success_oof["curve_pareto_frontier"]
            ]
        },
        "warning": (
            "This is not a cascade: it gates before execution, replays only one "
            "trajectory, and does not accumulate escalation cost."
        ),
    }


def run_vardanyan_style_escalation(
    outcomes: dict[int, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    """DOM-first trajectory, then Vision when an observable failure proxy fires."""

    n_success = 0
    total_cost = 0.0
    n_escalated = 0
    signal_counts: Counter[str] = Counter()
    final_mode_counts: Counter[str] = Counter()
    task_records: list[dict[str, Any]] = []
    for task_id in sorted(outcomes):
        summary_path = REPO / outcomes[task_id][VARDANYAN_START_MODE]["summary_path"]
        steps_path = _canonical_steps_path(summary_path)
        steps = read_jsonl_dedup(
            steps_path,
            summary_path=summary_path,
            strict_identity=True,
        )
        if not steps:
            raise ValueError(f"No DOM steps for Vardanyan-style task {task_id}")
        signals = trajectory_failure_signals(steps)
        for signal, fired in signals.items():
            signal_counts[signal] += int(fired)
        escalate = any(signals.values())
        row = simulate_two_stage_task(
            outcomes[task_id],
            start_mode=VARDANYAN_START_MODE,
            escalation_mode=VARDANYAN_ESCALATION_MODE,
            escalate=escalate,
        )
        n_success += int(row["success"])
        total_cost += float(row["total_billed_cost_usd"])
        n_escalated += int(escalate)
        final_mode_counts[row["final_mode"]] += 1
        task_records.append(
            {
                "task_id": task_id,
                "escalated": escalate,
                "signals": signals,
                "steps_path": _relative(steps_path),
                **row,
            }
        )
    point = _point_from_metric(
        "vardanyan_dom_to_vision_failure",
        "Vardanyan-style DOM→Vision failure escalation",
        "litmined_heuristic_escalation",
        {
            "n_tasks": len(outcomes),
            "n_success": n_success,
            "sum_total_billed_cost_usd": total_cost,
        },
    )
    return {
        "protocol": {
            "source_style": "vardanyan2025browser",
            "start_mode": VARDANYAN_START_MODE,
            "escalation_mode": VARDANYAN_ESCALATION_MODE,
            "decision_rule": (
                "escalate after any DOM step parse failure, failed/invalid action, "
                "two consecutive unchanged pages, or finish with an explicitly empty answer"
            ),
            "decision_uses_evaluator_success": False,
            "cost_accounting": (
                "sum complete independent-reset DOM and Vision Pass-1 trajectories "
                "when escalation fires"
            ),
            "success_accounting": "success of final executed mode only",
        },
        "point": {
            **point,
            "mean_trajectories_executed": 1.0 + n_escalated / len(outcomes),
            "escalated_task_fraction": n_escalated / len(outcomes),
            "n_escalated": n_escalated,
            "final_mode_counts": dict(sorted(final_mode_counts.items())),
            "signal_task_counts": dict(sorted(signal_counts.items())),
        },
        "task_records": task_records,
    }


def run_lazymcot_style_escalation(
    outcomes: dict[int, dict[str, dict[str, Any]]],
    entries: dict[str, dict[str, Any]],
    fold_map: dict[int, int],
) -> dict[str, Any]:
    """Vision-first, task-length difficulty trigger, then SoM re-observation."""

    raw_by_task = {
        task_id: build_offline_raw_features(
            entries, PHASE1_ROOT, "classifieds", task_id
        )[0]
        for task_id in sorted(outcomes)
    }
    fold_thresholds: dict[int, float] = {}
    for fold_k in range(N_FOLDS):
        train_lengths = [
            int(raw_by_task[task_id]["numeric"][3])
            for task_id in outcomes
            if fold_map[task_id] != fold_k
        ]
        fold_thresholds[fold_k] = _fold_train_quantile_threshold(
            train_lengths, LAZYMCOT_DIFFICULTY_QUANTILE
        )

    n_success = 0
    total_cost = 0.0
    n_escalated = 0
    final_mode_counts: Counter[str] = Counter()
    task_records: list[dict[str, Any]] = []
    for task_id in sorted(outcomes):
        fold_k = fold_map[task_id]
        intent_token_count = int(raw_by_task[task_id]["numeric"][3])
        threshold = fold_thresholds[fold_k]
        escalate = difficulty_trigger(intent_token_count, threshold)
        row = simulate_two_stage_task(
            outcomes[task_id],
            start_mode=LAZYMCOT_START_MODE,
            escalation_mode=LAZYMCOT_ESCALATION_MODE,
            escalate=escalate,
        )
        n_success += int(row["success"])
        total_cost += float(row["total_billed_cost_usd"])
        n_escalated += int(escalate)
        final_mode_counts[row["final_mode"]] += 1
        task_records.append(
            {
                "task_id": task_id,
                "fold_k": fold_k,
                "intent_token_count": intent_token_count,
                "fold_train_median_threshold": threshold,
                "escalated": escalate,
                **row,
            }
        )
    point = _point_from_metric(
        "lazymcot_length_median_vision_to_som",
        "LazyMCoT-style length trigger Vision→SoM",
        "litmined_difficulty_escalation",
        {
            "n_tasks": len(outcomes),
            "n_success": n_success,
            "sum_total_billed_cost_usd": total_cost,
        },
    )
    return {
        "protocol": {
            "source_style": "wang2026lazymcot",
            "outer_folds": N_FOLDS,
            "same_fold_map": True,
            "difficulty_proxy": "task intent whitespace-token count",
            "threshold": "strictly above held-out fold's training median",
            "threshold_quantile": LAZYMCOT_DIFFICULTY_QUANTILE,
            "fold_train_thresholds": {
                str(key): value for key, value in sorted(fold_thresholds.items())
            },
            "start_mode": LAZYMCOT_START_MODE,
            "escalation_mode": LAZYMCOT_ESCALATION_MODE,
            "decision_uses_evaluator_success": False,
            "cost_accounting": (
                "sum complete independent-reset Vision and SoM Pass-1 trajectories "
                "when escalation fires"
            ),
            "success_accounting": "success of final executed mode only",
        },
        "point": {
            **point,
            "mean_trajectories_executed": 1.0 + n_escalated / len(outcomes),
            "escalated_task_fraction": n_escalated / len(outcomes),
            "n_escalated": n_escalated,
            "final_mode_counts": dict(sorted(final_mode_counts.items())),
        },
        "task_records": task_records,
    }


def run_litmined_baselines(
    outcomes: dict[int, dict[str, dict[str, Any]]],
    entries: dict[str, dict[str, Any]],
    fold_map: dict[int, int],
) -> dict[str, Any]:
    vardanyan = run_vardanyan_style_escalation(outcomes)
    lazymcot = run_lazymcot_style_escalation(outcomes, entries, fold_map)
    return {
        "implemented_policy_count": 2,
        "vardanyan_style": vardanyan,
        "lazymcot_style": lazymcot,
        "points": [vardanyan["point"], lazymcot["point"]],
    }


def _random_summary(
    policy_id: str,
    label: str,
    sr_repetitions: np.ndarray,
    cost_repetitions: np.ndarray,
    *,
    n_tasks: int,
    best_sr: float,
    oracle_sr: float,
) -> dict[str, Any]:
    pgr = np.asarray(
        [performance_gap_recovered(value, best_sr, oracle_sr) for value in sr_repetitions],
        dtype=float,
    )
    sr_mean = float(np.mean(sr_repetitions))
    cost_mean = float(np.mean(cost_repetitions))
    point = asdict(
        PolicyPoint(
            policy_id=policy_id,
            label=label,
            category="random_noise_floor",
            mean_cost_usd=cost_mean,
            success_rate_pct=sr_mean,
            n_tasks=n_tasks,
            n_success=int(round(n_tasks * sr_mean / 100.0)),
        )
    )
    return {
        **point,
        "point_estimate_kind": "mean across repeated stochastic routers",
        "success_rate_pct_sd": float(np.std(sr_repetitions, ddof=1)),
        "mean_cost_usd_sd": float(np.std(cost_repetitions, ddof=1)),
        "performance_gap_recovered_mean": float(np.mean(pgr)),
        "performance_gap_recovered_sd": float(np.std(pgr, ddof=1)),
        "success_rate_pct_percentile_95": [
            float(value) for value in np.percentile(sr_repetitions, [2.5, 97.5])
        ],
        "mean_cost_usd_percentile_95": [
            float(value) for value in np.percentile(cost_repetitions, [2.5, 97.5])
        ],
    }


def run_random_noise_floors(
    outcomes: dict[int, dict[str, dict[str, Any]]],
    fold_map: dict[int, int],
    *,
    n_repetitions: int,
    seed: int,
    best_sr: float,
    oracle_sr: float,
) -> dict[str, Any]:
    task_ids = sorted(outcomes)
    success = np.asarray(
        [
            [float(outcomes[task_id][mode]["success"]) for mode in DISPLAY_MODES]
            for task_id in task_ids
        ]
    )
    cost = np.asarray(
        [
            [float(outcomes[task_id][mode]["cost_usd"]) for mode in DISPLAY_MODES]
            for task_id in task_ids
        ]
    )
    n_tasks, n_modes = success.shape

    uniform_rng = np.random.default_rng(seed)
    uniform_choice = uniform_rng.integers(0, n_modes, size=(n_repetitions, n_tasks))

    cost_context: dict[int, Any] = {}
    cost_probability = np.empty((n_tasks, n_modes), dtype=float)
    for fold_k in range(N_FOLDS):
        train_ids = [task_id for task_id in task_ids if fold_map[task_id] != fold_k]
        mean_cost = np.asarray(
            [
                np.mean([outcomes[task_id][mode]["cost_usd"] for task_id in train_ids])
                for mode in DISPLAY_MODES
            ],
            dtype=float,
        )
        inverse = 1.0 / mean_cost
        probability = inverse / inverse.sum()
        cost_context[fold_k] = {
            "mean_train_cost_usd_by_mode": {
                mode: float(value) for mode, value in zip(DISPLAY_MODES, mean_cost)
            },
            "inverse_cost_probability_by_mode": {
                mode: float(value) for mode, value in zip(DISPLAY_MODES, probability)
            },
        }
        for row_idx, task_id in enumerate(task_ids):
            if fold_map[task_id] == fold_k:
                cost_probability[row_idx] = probability

    weighted_rng = np.random.default_rng(seed + 1)
    weighted_choice = np.empty((n_repetitions, n_tasks), dtype=int)
    for task_idx in range(n_tasks):
        weighted_choice[:, task_idx] = weighted_rng.choice(
            n_modes,
            size=n_repetitions,
            replace=True,
            p=cost_probability[task_idx],
        )

    task_columns = np.arange(n_tasks)[None, :]

    def evaluate(choices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sr = 100.0 * success[task_columns, choices].mean(axis=1)
        costs = cost[task_columns, choices].mean(axis=1)
        return sr, costs

    uniform_sr, uniform_cost = evaluate(uniform_choice)
    weighted_sr, weighted_cost = evaluate(weighted_choice)
    return {
        "protocol": {
            "n_repetitions": n_repetitions,
            "uniform_seed": seed,
            "cost_weighted_seed": seed + 1,
            "cost_weight_definition": (
                "within each held-out fold, mode probability proportional to the "
                "inverse fold-train mean total_billed_cost_usd"
            ),
        },
        "fold_cost_weight_context": {str(key): value for key, value in cost_context.items()},
        "points": [
            _random_summary(
                "random_uniform",
                "Uniform random (1000-repeat mean)",
                uniform_sr,
                uniform_cost,
                n_tasks=n_tasks,
                best_sr=best_sr,
                oracle_sr=oracle_sr,
            ),
            _random_summary(
                "random_inverse_cost",
                "Inverse-cost random (1000-repeat mean)",
                weighted_sr,
                weighted_cost,
                n_tasks=n_tasks,
                best_sr=best_sr,
                oracle_sr=oracle_sr,
            ),
        ],
    }


def _policy_point(row: dict[str, Any]) -> PolicyPoint:
    return PolicyPoint(
        **{key: row[key] for key in PolicyPoint.__dataclass_fields__}
    )


def _compact_point(row: dict[str, Any]) -> dict[str, Any]:
    """Keep the combined plane light; detailed task records stay by family."""

    keys = [
        *PolicyPoint.__dataclass_fields__,
        "performance_gap_recovered",
        "delta_vs_best_single_sr_pp",
        "delta_vs_best_single_cost_usd",
        "delta_sr_pp_per_delta_usd",
        "point_estimate_kind",
        "success_rate_pct_sd",
        "mean_cost_usd_sd",
        "performance_gap_recovered_mean",
        "performance_gap_recovered_sd",
        "signal",
        "tau_quantile",
        "mean_trajectories_executed",
        "escalated_task_fraction",
        "n_escalated",
        "k",
    ]
    return {key: row[key] for key in keys if key in row}


def build_combined_pareto(points: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(points)
    ids = [str(row["policy_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        duplicates = sorted(policy_id for policy_id, count in Counter(ids).items() if count > 1)
        raise ValueError(f"Duplicate combined policy IDs: {duplicates}")
    deployable_rows = [row for row in rows if row["category"] != "hindsight_oracle"]
    oracle_rows = [row for row in rows if row["category"] == "hindsight_oracle"]
    if len(oracle_rows) != 1:
        raise ValueError("Combined Pareto set must contain exactly one oracle")
    deployable = [_policy_point(row) for row in deployable_rows]
    hindsight = deployable + [_policy_point(oracle_rows[0])]
    dominated_by: dict[str, list[dict[str, str]]] = {}
    for target in hindsight:
        dominators = []
        for candidate in hindsight:
            if candidate.policy_id == target.policy_id:
                continue
            relation = dominance_relation(candidate, target)
            if relation == "a_strictly_dominates_b":
                dominators.append({"policy_id": candidate.policy_id, "type": "strict"})
            elif relation == "a_weakly_dominates_b":
                dominators.append({"policy_id": candidate.policy_id, "type": "weak"})
        dominated_by[target.policy_id] = dominators
    return {
        "deployable_frontier": [point.policy_id for point in pareto_frontier(deployable)],
        "hindsight_augmented_frontier": [
            point.policy_id for point in pareto_frontier(hindsight)
        ],
        "dominated_by": dominated_by,
        "n_deployable_points": len(deployable),
        "n_points_with_oracle": len(hindsight),
    }


def _best_row(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    return min(
        rows,
        key=lambda row: (-float(row["success_rate_pct"]), float(row["mean_cost_usd"]), row["policy_id"]),
    )


def _format_pgr(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def _format_ratio(value: float | None) -> str:
    return "—" if value is None else f"{value:+.1f}"


def _table_row(row: dict[str, Any], frontier: set[str]) -> str:
    sr_sd = row.get("success_rate_pct_sd")
    cost_sd = row.get("mean_cost_usd_sd")
    sr = f"{row['success_rate_pct']:.2f}" + ("" if sr_sd is None else f" ± {sr_sd:.2f}")
    cost = f"{row['mean_cost_usd']:.6f}" + ("" if cost_sd is None else f" ± {cost_sd:.6f}")
    pgr = _format_pgr(row.get("performance_gap_recovered"))
    if row.get("performance_gap_recovered_sd") is not None:
        pgr += f" ± {row['performance_gap_recovered_sd']:.3f}"
    return (
        f"| {row['label']} | {sr} | {cost} | "
        f"{pgr} | "
        f"{_format_ratio(row.get('delta_sr_pp_per_delta_usd'))} | "
        f"{'yes' if row['policy_id'] in frontier else 'no'} |"
    )


def validate_literature_sources(
    rows: Iterable[tuple[str, str, str, str]] = LITERATURE_MINING_ROWS,
) -> None:
    """Fail closed if any cited local file or inclusive line range drifts."""

    line_counts: dict[Path, int] = {}
    for work, _, _, source_text in rows:
        chunks = source_text.split("; docs/")
        references = [chunks[0], *(f"docs/{chunk}" for chunk in chunks[1:])]
        for reference in references:
            try:
                relative_path, span = reference.rsplit(":", 1)
                start, end = (int(value) for value in span.split("-", 1))
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"Malformed literature source for {work}: {reference}"
                ) from exc
            path = REPO / relative_path
            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing literature source for {work}: {relative_path}"
                )
            if path not in line_counts:
                line_counts[path] = len(path.read_text().splitlines())
            if start < 1 or end < start or end > line_counts[path]:
                raise ValueError(
                    f"Invalid literature line range for {work}: {reference} "
                    f"(file has {line_counts[path]} lines)"
                )


def literature_mining_inventory() -> list[dict[str, str]]:
    """Return the validated local-source inventory in a report-friendly form."""

    validate_literature_sources()
    return [
        {
            "work": work,
            "routed_object": routed_object,
            "offline_adaptability": adaptability,
            "local_source": source,
        }
        for work, routed_object, adaptability, source in LITERATURE_MINING_ROWS
    ]


def _markdown_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def render_literature_inventory(rows: Iterable[dict[str, str]]) -> list[str]:
    lines = [
        "## Systematic router/switch mining inventory",
        "",
        "Local-source-only inventory; `UNVERIFIED` is preserved wherever the allowed "
        "notes do not support a stronger claim. “能但不新增” means the idea honestly "
        "maps to this replay but collapses to an existing fixed/analysis point.",
        "",
        "| Prior work | Routed object | Six-mode task-level offline replay adaptation | Local source |",
        "|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                _markdown_cell(row[key])
                for key in (
                    "work",
                    "routed_object",
                    "offline_adaptability",
                    "local_source",
                )
            )
            + " |"
        )
    return lines


def render_markdown(payload: dict[str, Any], *, report_title: str | None = None) -> str:
    title = report_title or "Router prior-work baselines — offline efficiency suite"
    points = {row["policy_id"]: row for row in payload["points"]}
    frontier = set(payload["pareto"]["deployable_frontier"])
    primary_signal = payload["cascade"]["primary_signal"]
    primary_curve = payload["cascade"]["curves"][primary_signal]
    knn_rows = payload["knn"]["points"] or [
        row for row in points.values() if row["category"] == "prior_knn"
    ]
    random_rows = payload["random_noise_floors"]["points"] or [
        row for row in points.values() if row["category"] == "random_noise_floor"
    ]
    cascade_best = [_best_row(rows) for rows in payload["cascade"]["curves"].values()]
    litmined_rows = payload["litmined_baselines"]["points"]
    best_knn = _best_row(knn_rows)
    best_cascade = _best_row(
        row for rows in payload["cascade"]["curves"].values() for row in rows
    )
    summary_rows = [
        points["fixed_som"],
        points["router_oof"],
        points["oracle"],
        *knn_rows,
        *litmined_rows,
        *random_rows,
        *cascade_best,
    ]
    lines = [
        f"# {DISCLAIMER}",
        "",
        f"## {title}",
        "",
        "All results are B0-Classifieds only (n=224), reuse the locked five-fold map, "
        "and use canonical episode-level `total_billed_cost_usd`. PGR is the requested "
        "offline adaptation `(SR_policy − SR_best-single)/(SR_oracle − SR_best-single)`. "
        "Random rows show 1000-repeat mean ± sample SD.",
        "",
    ]
    lines.extend(render_literature_inventory(payload["literature_mining_inventory"]))
    lines.extend(
        [
            "",
            "## Main operating points",
            "",
            "| Policy | SR (%) | Mean billed USD/task | PGR | ΔSR pp / ΔUSD vs best-single | Deployable frontier |",
            "|---|---:|---:|---:|---:|:---:|",
        ]
    )
    lines.extend(_table_row(row, frontier) for row in summary_rows)
    lines.extend(
        [
            "",
            "The signed ΔSR/ΔUSD column is descriptive, not a scalar ranking: when both "
            "deltas are negative, a positive value means *SR lost per dollar saved*. Relative "
            f"to the locked LR, the best kNN point (`{best_knn['policy_id']}`) raises SR by "
            f"{best_knn['success_rate_pct'] - points['router_oof']['success_rate_pct']:+.2f} pp "
            f"and changes cost by ${best_knn['mean_cost_usd'] - points['router_oof']['mean_cost_usd']:+.6f}/task, "
            "so it strictly dominates locked LR. It is itself strictly dominated by the "
            "existing post-hoc OOF P(success) τ=0.05 and τ=0.10 points. The highest-SR "
            f"cascade point (`{best_cascade['policy_id']}`) reaches only "
            f"{best_cascade['success_rate_pct']:.2f}% at ${best_cascade['mean_cost_usd']:.6f}/task; "
            "it recovers none of the best-single-to-oracle gap.",
            "",
            "## FrugalGPT-style threshold curve (primary observed signal)",
            "",
            f"Primary signal: `{primary_signal}`. Here q indexes a threshold computed only "
            "from that held-out fold's training task×mode confidence values. q=0 executes "
            "the fold-cheapest mode only; q=1 forces the full six-trajectory chain.",
            "",
            "| q | SR (%) | Mean billed USD/task | PGR | Mean trajectories | Escalated tasks | Frontier |",
            "|---:|---:|---:|---:|---:|---:|:---:|",
        ]
    )
    for row in primary_curve:
        lines.append(
            f"| {row['tau_quantile']:.2f} | {row['success_rate_pct']:.2f} | "
            f"{row['mean_cost_usd']:.6f} | {_format_pgr(row.get('performance_gap_recovered'))} | "
            f"{row['mean_trajectories_executed']:.2f} | "
            f"{row['escalated_task_fraction']:.1%} | "
            f"{'yes' if row['policy_id'] in frontier else 'no'} |"
        )

    lines.extend(
        [
            "",
            "## Newly mined web/observation-router adaptations",
            "",
            "Both policies use only pre-evaluator decision signals and sum the full "
            "canonical billed cost of every replayed trajectory. Because Pass-1 arms "
            "were run from independent resets, these are conservative two-trajectory "
            "accounting simulations, not live state hand-offs.",
            "",
            "| Policy | SR (%) | Mean billed USD/task | PGR | Escalated tasks | Mean trajectories | Frontier |",
            "|---|---:|---:|---:|---:|---:|:---:|",
        ]
    )
    for row in litmined_rows:
        lines.append(
            f"| {row['label']} | {row['success_rate_pct']:.2f} | "
            f"{row['mean_cost_usd']:.6f} | "
            f"{_format_pgr(row.get('performance_gap_recovered'))} | "
            f"{row['n_escalated']} ({row['escalated_task_fraction']:.1%}) | "
            f"{row['mean_trajectories_executed']:.2f} | "
            f"{'yes' if row['policy_id'] in frontier else 'no'} |"
        )

    lines.extend(["", "## Dominance conclusion", ""])
    deployable_labels = [points[policy_id]["label"] for policy_id in payload["pareto"]["deployable_frontier"]]
    hindsight_labels = [points[policy_id]["label"] for policy_id in payload["pareto"]["hindsight_augmented_frontier"]]
    lines.append(f"Deployable frontier after adding every point: **{', '.join(deployable_labels)}**.")
    lines.append(f"With the hindsight oracle included: **{', '.join(hindsight_labels)}**.")
    for policy_id in [
        "router_oof",
        *(row["policy_id"] for row in knn_rows),
        *(row["policy_id"] for row in litmined_rows),
    ]:
        dominators = payload["pareto"]["dominated_by"][policy_id]
        text = ", ".join(f"{row['policy_id']} ({row['type']})" for row in dominators) or "none"
        lines.append(f"- `{policy_id}` dominators: {text}.")

    probe = payload["confidence_probe"]
    lines.extend(
        [
            "",
            "## Confidence probe",
            "",
            f"Episode summary JSON contains no embedded confidence aggregate. Canonical step "
            f"JSONL supplies the fields: {probe['n_steps_with_confidence_dict']}/{probe['n_steps']} "
            f"steps ({probe['step_confidence_coverage']:.1%}) and all "
            f"{probe['n_task_mode_episodes']} task×mode episodes have complete "
            f"`mean/min_logprob` and `mean/min_margin` aggregates. Therefore the reported "
            f"cascade is the observed post-trajectory version, not the OOF P(success) fallback.",
            "",
            "## Deviations from the cited methods",
            "",
            "- **RouteLLM-style, not RouteLLM reproduction (`ong2025routellm`).** RouteLLM's "
            "similarity-weighted router uses query embeddings and a locally weighted "
            "Bradley–Terry preference model for binary strong/weak routing. Here we follow the "
            "advisor-specified simpler kNN adaptation: fold-local TF-IDF/MI-18, cosine kNN, "
            "uniform majority vote over six-mode cheapest-success oracle labels. There is no "
            "human preference data, exponential similarity weighting, BT fit, or binary call threshold.",
            "- **FrugalGPT-style, not FrugalGPT reproduction (`chen2023frugalgpt`).** Modes are "
            "ordered by fold-training realized billed cost; escalation uses aggregated token "
            "logprob/margin rather than a learned response scorer or cascade optimizer. Each "
            "Pass-1 trajectory originally starts from an independent reset; summing them is an "
            "offline accounting simulation, not a live stateful cascade.",
            "- **Vardanyan-style, not production-report reproduction (`vardanyan2025browser`).** "
            "The report's documented triggers are canvas apps, charts, or a missing accessibility "
            "tree. Those page-semantic triggers are not logged in the replay, so the adaptation "
            "uses outcome-free step failures (parse/action/unchanged-page/explicit empty finish) "
            "and escalates a complete DOM trajectory to a complete Vision trajectory.",
            "- **LazyMCoT-style, not LazyMCoT reproduction (`wang2026lazymcot`).** The source uses "
            "first-token statistics, a small GBDT, conformal thresholding, and localized-panel "
            "re-observation in natural-image VQA. The adaptation uses only task-text length, a "
            "fold-training median threshold, and a complete Vision→SoM replay on VWA.",
            "- **WebRouter not implemented (`webrouter2025`).** The allowed local sources expose "
            "only the ca-VIB name and query-to-model scope, not a reproducible objective; a routing "
            "scan also conflicts with the verified bibliography by calling its object compressed "
            "prompts. Implementing a made-up cost regularizer would violate the local-source-only rule.",
            "- **Noise floors are study-specific controls.** Uniform random and inverse-fold-train-cost "
            "random are not claimed as implementations from either cited paper.",
            "- **PGR is an offline adaptation.** RouteLLM normalizes between weak and strong models; "
            "the requested suite substitutes the best fixed representation and the six-mode "
            "hindsight oracle. PGR may therefore be negative and the oracle is exactly 1.",
            "- **APGR is not transferred.** RouteLLM integrates PGR over a binary strong-model-call "
            "rate. A six-arm representation menu with endogenous dollar costs—and cascades that "
            "can execute multiple arms—has no faithful strong-call axis. Per the directive, we "
            "report pointwise PGR and signed ΔSR/ΔUSD instead of inventing an APGR analogue.",
            "",
            "## Scope caveats",
            "",
            "- OFFLINE/NON-GATE/post-hoc exploratory; B0-Classifieds is the only complete OOF cell.",
            "- kNN and fixed/random policies replay one trajectory per task. Cascade policies "
            "sum all visited trajectories and use only the final mode's success.",
            "- Router overhead, fresh serving stochasticity, latency, and sequential environment "
            "mutation are not represented. No cross-cell cost pooling is performed.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_analysis(
    *,
    replay_path: Path,
    pareto_path: Path,
    success_oof_path: Path,
    source_dir: Path,
    random_repetitions: int,
    random_seed: int,
) -> dict[str, Any]:
    replay = json.loads(replay_path.read_text())
    pareto_payload = json.loads(pareto_path.read_text())
    success_oof = json.loads(success_oof_path.read_text())
    if replay.get("artifact_status") != "OFFLINE/NON-GATE" or replay.get("gate_eligible") is not False:
        raise ValueError("Replay input must be OFFLINE/NON-GATE and gate_eligible=false")
    replay_cell = replay["cells"][CELL_ID]
    if replay_cell.get("training_status") != "TRAINED_COMPLETE":
        raise ValueError(f"{CELL_ID} is not a full-OOF cell")
    fold_path = source_dir / f"{CELL_ID}_fold_assignment.json"
    fold_payload = json.loads(fold_path.read_text())
    fold_map = {
        int(task_id): int(fold) for task_id, fold in fold_payload["fold_assignment"].items()
    }
    fold_sha = fold_map_sha256(fold_payload["fold_assignment"])
    if fold_sha != replay_cell["fold_map_sha256"]:
        raise ValueError("Fold-map SHA drift against offline replay")
    if fold_sha != success_oof["protocol"]["fold_map_sha256"]:
        raise ValueError("Fold-map SHA drift against OOF success probabilities")

    manifest_path = REPO / replay["inputs"]["run_manifest"]
    entries = load_paper_grade_entries(manifest_path, [CELL_ID])[CELL_ID]
    outcomes, outcome_provenance = collect_cell_outcomes(
        CELL_ID, entries, PHASE1_ROOT
    )
    if set(fold_map) != set(outcomes) or len(outcomes) != int(replay_cell["n_tasks"]):
        raise ValueError("Fold map, task universe, and replay coverage are not identical")
    if (
        outcome_provenance["canonical_task_universe_sha256"]
        != replay_cell["outcome_provenance"]["canonical_task_universe_sha256"]
    ):
        raise ValueError("Canonical task-universe SHA drift")

    base_points = pareto_payload["cells"][CELL_ID]["points"]
    base_by_id = {row["policy_id"]: dict(row) for row in base_points}
    required = {f"fixed_{mode}" for mode in DISPLAY_MODES} | {"router_oof", "oracle"}
    if set(base_by_id) != required:
        raise ValueError(f"Unexpected base Pareto point set: {sorted(base_by_id)}")
    best = base_by_id[pareto_payload["cells"][CELL_ID]["best_single_policy_id"]]
    oracle = base_by_id["oracle"]
    best_sr = float(best["success_rate_pct"])
    best_cost = float(best["mean_cost_usd"])
    oracle_sr = float(oracle["success_rate_pct"])

    knn = run_knn_baselines(outcomes, entries, fold_map, source_dir)
    confidence, confidence_probe = load_observed_confidence(outcomes)
    if confidence_probe["complete_observed_signals"]:
        cascade = run_observed_cascades(
            outcomes,
            confidence,
            confidence_probe["complete_observed_signals"],
            fold_map,
        )
    else:
        cascade = build_preexecution_fallback(success_oof)
    litmined = run_litmined_baselines(outcomes, entries, fold_map)
    random_floors = run_random_noise_floors(
        outcomes,
        fold_map,
        n_repetitions=random_repetitions,
        seed=random_seed,
        best_sr=best_sr,
        oracle_sr=oracle_sr,
    )

    cost_aware_points = [dict(row) for row in success_oof["curve"]]
    all_points = [
        *[dict(row) for row in base_points],
        *cost_aware_points,
        *[dict(row) for row in knn["points"]],
        *[
            dict(row)
            for curve in cascade["curves"].values()
            for row in curve
        ],
        *[dict(row) for row in litmined["points"]],
        *[dict(row) for row in random_floors["points"]],
    ]
    all_points = [
        _attach_efficiency(
            row,
            best_sr=best_sr,
            best_cost=best_cost,
            oracle_sr=oracle_sr,
        )
        for row in all_points
    ]
    by_id = {row["policy_id"]: row for row in all_points}
    knn["points"] = [by_id[row["policy_id"]] for row in knn["points"]]
    random_floors["points"] = [
        {
            **by_id[row["policy_id"]],
            "performance_gap_recovered_mean": row["performance_gap_recovered_mean"],
            "performance_gap_recovered_sd": row["performance_gap_recovered_sd"],
        }
        for row in random_floors["points"]
    ]
    cascade["curves"] = {
        signal: [by_id[row["policy_id"]] for row in rows]
        for signal, rows in cascade["curves"].items()
    }
    litmined["points"] = [by_id[row["policy_id"]] for row in litmined["points"]]
    litmined["vardanyan_style"]["point"] = by_id[
        litmined["vardanyan_style"]["point"]["policy_id"]
    ]
    litmined["lazymcot_style"]["point"] = by_id[
        litmined["lazymcot_style"]["point"]["policy_id"]
    ]
    plane_points = [_compact_point(row) for row in all_points]
    pareto = build_combined_pareto(plane_points)
    best_knn = _best_row(knn["points"])
    best_cascade = _best_row(
        row for rows in cascade["curves"].values() for row in rows
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_status": ARTIFACT_STATUS,
        "disclaimer": DISCLAIMER,
        "gate_eligible": False,
        "h10_eligible": False,
        "post_hoc_exploratory": True,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cell_id": CELL_ID,
        "n_tasks": len(outcomes),
        "definitions": {
            "cost": "mean/sum canonical Pass-1 total_billed_cost_usd per task",
            "pgr": "(SR_policy - SR_best_single) / (SR_oracle - SR_best_single)",
            "cost_normalized": "(SR_policy - SR_best_single) in percentage points divided by (cost_policy - cost_best_single) in USD/task",
            "strict_dominance": "lower cost AND higher SR",
            "weak_dominance": "no worse on both axes and strictly better on exactly one",
            "random_point": "mean operating point over fixed-seed repeated stochastic routers",
            "apgr": (
                "not computed: RouteLLM APGR integrates over binary strong-model-call rate, "
                "which has no faithful analogue for a six-arm dollar-cost menu with cumulative cascades"
            ),
        },
        "protocol": {
            "outer_folds": N_FOLDS,
            "same_task_universe": True,
            "same_fold_map": True,
            "fold_map_sha256": fold_sha,
            "feature_assets": "locked fold-local TF-IDF + MI-18",
            "cost_unit_basis": outcome_provenance["cost_unit_basis"],
            "cost_field": "total_billed_cost_usd",
        },
        "inputs": {
            "replay_json": _relative(replay_path),
            "replay_json_sha256": sha256_file(replay_path),
            "pareto_json": _relative(pareto_path),
            "pareto_json_sha256": sha256_file(pareto_path),
            "cost_aware_success_oof_json": _relative(success_oof_path),
            "cost_aware_success_oof_json_sha256": sha256_file(success_oof_path),
            "fold_map": _relative(fold_path),
            "fold_map_file_sha256": sha256_file(fold_path),
            "fold_assignment_content_sha256": fold_sha,
            "canonical_task_universe_sha256": outcome_provenance[
                "canonical_task_universe_sha256"
            ],
            "run_manifest": _relative(manifest_path),
            "run_manifest_sha256": sha256_file(manifest_path),
        },
        "literature_adaptation": {
            "ong2025routellm": (
                "advisor-specified kNN simplification in locked TF-IDF/MI-18 space; "
                "not RouteLLM's embedding-based exponentially similarity-weighted "
                "Bradley-Terry preference router"
            ),
            "chen2023frugalgpt": (
                "fold-cost-ordered six-mode trajectory cascade using observed B0 "
                "logprob/margin aggregates; not FrugalGPT's learned cascade selection/scoring"
            ),
            "vardanyan2025browser": (
                "DOM-first then Vision on logged parse/action/no-progress proxies; "
                "the report's production triggers are canvas/charts/missing a11y, and "
                "offline replay must sum independent full trajectories"
            ),
            "wang2026lazymcot": (
                "Vision-first then SoM when task-text length exceeds the fold-training "
                "median; substitutes a static intent-length proxy for first-token "
                "statistics and full SoM for localized-panel re-observation"
            ),
            "webrouter2025": (
                "not implemented: local sources identify a cost-aware variational "
                "information-bottleneck query-to-model router but do not expose an "
                "exact objective or training recipe; one routing note also conflicts "
                "with the verified bibliography on the routed object"
            ),
        },
        "literature_mining_inventory": literature_mining_inventory(),
        "anchors": {
            "best_single_policy_id": best["policy_id"],
            "best_single_success_rate_pct": best_sr,
            "best_single_mean_cost_usd": best_cost,
            "oracle_policy_id": "oracle",
            "oracle_success_rate_pct": oracle_sr,
            "oracle_mean_cost_usd": float(oracle["mean_cost_usd"]),
            "pgr_denominator_pp": oracle_sr - best_sr,
        },
        "confidence_probe": confidence_probe,
        "knn": knn,
        "cascade": cascade,
        "litmined_baselines": litmined,
        "random_noise_floors": random_floors,
        "pareto": pareto,
        "summary": {
            "best_knn_policy_id": best_knn["policy_id"],
            "best_knn_success_rate_pct": best_knn["success_rate_pct"],
            "best_knn_mean_cost_usd": best_knn["mean_cost_usd"],
            "best_cascade_policy_id": best_cascade["policy_id"],
            "best_cascade_success_rate_pct": best_cascade["success_rate_pct"],
            "best_cascade_mean_cost_usd": best_cascade["mean_cost_usd"],
            "primary_cascade_signal": cascade["primary_signal"],
            "litmined_policy_ids": [
                row["policy_id"] for row in litmined["points"]
            ],
            "locked_lr_on_combined_deployable_frontier": "router_oof" in pareto[
                "deployable_frontier"
            ],
        },
        "points": plane_points,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-json", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--pareto-json", type=Path, default=DEFAULT_PARETO)
    parser.add_argument("--success-oof-json", type=Path, default=DEFAULT_SUCCESS_OOF)
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--random-repetitions", type=int, default=RANDOM_REPETITIONS)
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    replay_path = args.replay_json.resolve()
    pareto_path = args.pareto_json.resolve()
    success_oof_path = args.success_oof_json.resolve()
    source_dir = args.source_dir.resolve()
    out_dir = args.out_dir.resolve()
    report_path = args.report.resolve()
    for path in (replay_path, pareto_path, success_oof_path):
        if not path.is_file():
            parser.error(f"Required input not found: {path}")
    if args.random_repetitions != 1_000:
        parser.error("The reporting contract requires exactly 1000 random repetitions")
    if out_dir == FORBIDDEN_CANONICAL_OUT or FORBIDDEN_CANONICAL_OUT in out_dir.parents:
        parser.error("Refusing to write prior-work baselines into canonical l1_router")

    payload = run_analysis(
        replay_path=replay_path,
        pareto_path=pareto_path,
        success_oof_path=success_oof_path,
        source_dir=source_dir,
        random_repetitions=args.random_repetitions,
        random_seed=args.random_seed,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "router_prior_baselines.json"
    md_path = out_dir / "router_prior_baselines.md"
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    md_path.write_text(render_markdown(payload))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        render_markdown(payload, report_title="Codex execution report — router lit-mined baselines")
    )
    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")
    print(f"Wrote: {report_path}")
    print(DISCLAIMER)
    return 0


if __name__ == "__main__":
    sys.exit(main())
