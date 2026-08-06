"""B-1966 — 页面截图契约：哪些 mode 的模型输入带图，以及它不许悄悄漂移。

这个 bug 的形状值得记住：它**没有任何症状**。曲线、峰值、两个控制组一应俱全，
只是测的不是声称的那个对象。24 个 mechanistic cell 里，唯一暴露它的是两个
cell 的 per_task 结果逐位相同 —— 而那两个 cell 之所以可比，纯属实验设计的偶然
（`p4_som_ptext` 与 `p2_psom_ptext` 恰好只差 `--source-mode` 一个参数）。

所以这里的测试目标不是「验证当前值对不对」，而是**让下一次漂移必须撞上断言，
而不是等一次 24 选 2 的巧合**。三层：

1. 常量与 `apply_som` 实际返回的 `marked_image` 一致（防常量与分支各走各的）；
2. `som` 与 `phantom_som` 的模型输入必须**可区分**（这条正是 B-1966 违反的）；
3. 页面截图与任务参考图是两回事，不许混为一谈。
"""
from __future__ import annotations

import inspect

import pytest

from p79.experiment.som import (
    KNOWN_OBSERVATION_MODES,
    MODES_RECEIVING_PAGE_IMAGE,
    mode_receives_page_image,
)


# --------------------------------------------------------------------------- #
# 1. 常量 vs 实现：两者不许各走各的
# --------------------------------------------------------------------------- #

def test_every_known_mode_has_a_declared_image_answer():
    """未知 mode 必须抛错而不是默认 False。

    默认 False 正是 B-1966 能活下来的土壤：一个 typo 会安静地回答「不带图」，
    与真正的 phantom mode 无法分辨。
    """
    for mode in KNOWN_OBSERVATION_MODES:
        assert isinstance(mode_receives_page_image(mode), bool)

    with pytest.raises(ValueError, match="unknown observation mode"):
        mode_receives_page_image("phantum_som")   # 真实世界的 typo 形态
    with pytest.raises(ValueError):
        mode_receives_page_image("")


def test_declared_set_matches_apply_som_source():
    """常量必须与 `apply_som` 分支体里实际写的 `marked_image` 对齐。

    读源码而不是跑 `apply_som`：后者需要一个真实 obs（截图 + AXTree + 浏览器
    节点信息），在单元测试里造一个够真的 obs 反而会把断言变成对 mock 的断言。
    这里断言的是「分支体里那几行字面量」，那正是 B-1966 的所在层。
    """
    from p79.experiment import som as som_mod

    src = inspect.getsource(som_mod.apply_som if hasattr(som_mod, "apply_som") else som_mod)

    # phantom 家族 + dom：源码里必须出现 marked_image=None 的分支
    assert "marked_image=None" in src, (
        "apply_som 里找不到 marked_image=None —— 分支结构变了，"
        "MODES_RECEIVING_PAGE_IMAGE 需要重新核对"
    )
    # 而 som / vision 必须有非 None 的图来源
    assert ("marked_image=image" in src) or ("_build_som_result" in src), (
        "apply_som 里找不到给 som/vision 传图的路径"
    )

    assert MODES_RECEIVING_PAGE_IMAGE == frozenset({"som", "vision"}), (
        "页面截图契约变了。这不是可以顺手改的常量：它决定 phantom 家族到底还是不是 "
        "phantom（P-SoM 的定义就是 SoM prompt 减去那张图）。改它请同时更新 "
        "docs/reference/master_bug_catalog.md B-1966 与 som.py 的 docstring。"
    )


@pytest.mark.parametrize("mode,expected", [
    ("som", True),
    ("vision", True),
    ("dom", False),
    ("phantom_som", False),
    ("phantom_dom", False),
    ("phantom_text", False),
    ("phantom_prompt", False),
])
def test_per_mode_page_image_contract(mode, expected):
    assert mode_receives_page_image(mode) is expected


# --------------------------------------------------------------------------- #
# 2. 承重断言：som 与 phantom_som 必须可区分
# --------------------------------------------------------------------------- #

def test_som_and_phantom_som_differ_only_by_the_image():
    """B-1966 的核心断言。

    这两个 mode 共享 [SOM_MARKS] 文本 payload，**且 system prompt 逐字节相同**
    （设计如此 —— P-SoM 就是 SoM prompt 减去图）。于是那张图是它们**唯一**的
    区别。任何把图也统一掉的代码路径，都会让两者完全等价 —— 这正是 mechanistic
    patching 脚本曾经做的事。

    实测量级：B1·classifieds 224/224 个 task，som 的 step_0 prompt tokens 全部
    高于 phantom_som，配对差中位数 578 token。
    """
    from p79.agents._shared_vl_utils import build_mode_prompt_dispatch_table

    prompts = build_mode_prompt_dispatch_table()
    assert prompts["som"] == prompts["phantom_som"], (
        "som 与 phantom_som 的 system prompt 不再相同。若这是有意的改动，"
        "phantom_som 的定义（'prompt promises a screenshot, agent gets none'）"
        "就变了，paper 的 P-SoM 论证需要重写"
    )

    # prompt 相同 ⇒ 图必须不同，否则两个 mode 无法分辨
    assert mode_receives_page_image("som") != mode_receives_page_image("phantom_som"), (
        "som 与 phantom_som 在 prompt 相同的前提下，页面截图也相同 —— "
        "两个 mode 完全不可区分。这就是 B-1966"
    )


def test_patching_pilot_gates_both_sides_on_mode():
    """mechanistic patching 脚本必须从 som.py 派生图的有无，而不是自己写死。

    读源码断言，因为跑那个脚本需要 GPU + 模型 + 落盘 artifacts。要防的是
    「有人为了省事把 `mode_receives_page_image` 换回字面量」这类回退。
    """
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    src = (repo / "scripts" / "mechanistic" /
           "run_stage2b_continuation_pilot.py").read_text(encoding="utf-8")

    assert "mode_receives_page_image" in src, (
        "run_stage2b_continuation_pilot.py 不再从 som.py 派生页面截图契约 —— B-1966 回退风险"
    )
    # 两侧都必须过 gate；曾经 target 侧是写死的 None
    assert src.count("mode_receives_page_image(args.") >= 2, (
        "source 与 target 两侧都必须按各自的 mode 判定是否传图；"
        "只 gate 一侧会让另一侧成为下一个 B-1966"
    )


# --------------------------------------------------------------------------- #
# 3. 别把两种「图」混为一谈
# --------------------------------------------------------------------------- #

def test_page_image_is_not_task_reference_image():
    """任务参考图（task config 的 `image`）与页面截图是两条独立路径。

    参考图对**每个** mode 都发送（agent 里单独注入），页面截图只给 som/vision。
    把两者混淆会得出「所有 mode 都传图，所以 patching 传图没错」的结论 ——
    这正是 B-1966 被质疑时需要分清的那一层。
    """
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    agent_src = (repo / "p79" / "agents" / "qwen3vl_agent.py").read_text(encoding="utf-8")

    assert "reference_images" in agent_src, "参考图注入路径不见了，本测试的前提需要重核"

    # 参考图的注入不得被 observation mode 门控 —— 它对所有 mode 都发
    ref_idx = agent_src.index("if reference_images:")
    window = agent_src[max(0, ref_idx - 400):ref_idx]
    assert "mode_receives_page_image" not in window, (
        "任务参考图的注入被页面截图契约门控了 —— 两条路径被混在一起"
    )
