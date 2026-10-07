"""B-2002 detector: the URL before a step is the previous step's obs_url (start_url at step 0).

Catches: reading the URL before an action from the same step's obs_url (which is the URL AFTER
it — every TYPE would then compare a query with itself), counting a TYPE into an empty box as
"filled", and flagging a correct replacement as an append.
"""
import json

from scripts.analysis.lib.shopping_contamination import episode_b2002

BASE = "http://localhost:7770/catalogsearch/result/?q="


def _write(tmp_path, steps):
    p = tmp_path / "shopping_task_1_steps_v2.jsonl"
    p.write_text("\n".join(json.dumps(s) for s in steps) + "\n", encoding="utf-8")
    return p


def test_append_replace_and_empty_box(tmp_path):
    steps = [
        # step 0: box pre-filled by start_url with "chair"; submit appends -> defect
        {"step_idx": 0, "action": {"action_type": "type", "text": "chair with wheels\n"},
         "obs_url": BASE + "chairchair+with+wheels"},
        # step 1: navigate away to a page with no query
        {"step_idx": 1, "action": {"action_type": "click"}, "obs_url": "http://localhost:7770/"},
        # step 2: type into an empty box -> not "filled", not counted
        {"step_idx": 2, "action": {"action_type": "type", "text": "lamp"}, "obs_url": BASE + "lamp"},
        # step 3: box holds "lamp"; correct replacement -> filled, not appended
        {"step_idx": 3, "action": {"action_type": "type", "text": "desk lamp"}, "obs_url": BASE + "desk+lamp"},
    ]
    r = episode_b2002(_write(tmp_path, steps), start_url=BASE + "chair")
    assert r == {"typed_into_filled": 2, "appended": 1}
