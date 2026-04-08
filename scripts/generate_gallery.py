#!/usr/bin/env python3
"""Generate an HTML gallery of annotated (or raw) screenshots for easy browsing.

Usage:
    # Single task
    python3 scripts/generate_gallery.py --run-dir results/.../B1_xxx --task-id 17

    # Single condition, all tasks
    python3 scripts/generate_gallery.py --run-dir results/.../B1_xxx --condition phase1_dom_router_0

    # Entire run
    python3 scripts/generate_gallery.py --run-dir results/.../B1_xxx

Output: <run-dir>/gallery.html (open in browser)
"""
from __future__ import annotations

import argparse
import base64
import json
import html as html_mod
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _read_steps(jsonl_path: Path) -> List[Dict[str, Any]]:
    """Read step JSONL with dedup."""
    try:
        from p79.experiment.io_utils import read_jsonl_dedup
        return read_jsonl_dedup(jsonl_path)
    except ImportError:
        lines = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    lines.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return lines


def _img_to_data_uri(img_path: Path) -> Optional[str]:
    """Convert image to base64 data URI for embedding in HTML."""
    if not img_path.exists():
        return None
    data = img_path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _img_to_relative(img_path: Path, gallery_path: Path) -> Optional[str]:
    """Get relative path from gallery HTML to image."""
    if not img_path.exists():
        return None
    try:
        return str(img_path.relative_to(gallery_path.parent))
    except ValueError:
        return str(img_path)


# ---------------------------------------------------------------------------
# Action summary (adapted from annotate_screenshots._build_action_line)
# ---------------------------------------------------------------------------

def _build_action_summary(step: Dict[str, Any]) -> str:
    """Build a compact action summary string from a step record."""
    action = step.get("action", {})
    if isinstance(action, str):
        try:
            action = json.loads(action)
        except (json.JSONDecodeError, TypeError):
            return action[:120] if action else ""

    action_type = str(
        step.get("action_type") or action.get("action_type") or "unknown"
    ).lower()

    parts = [action_type.upper()]
    eid = action.get("element_id")
    coord = action.get("coordinate")

    if eid is not None:
        parts.append(f"[{eid}]")

    if coord and isinstance(coord, (list, tuple)) and len(coord) >= 2:
        parts.append(f"@({coord[0]:.2f},{coord[1]:.2f})")

    if action_type == "type":
        text = str(action.get("text", "")).replace("\n", "\\n")
        if len(text) > 40:
            text = text[:37] + "..."
        parts.append(f"'{text}'")
    elif action_type == "finish":
        answer = str(action.get("answer", ""))
        if len(answer) > 40:
            answer = answer[:37] + "..."
        parts.append(f"ans='{answer}'")
    elif action_type == "scroll":
        delta = action.get("delta", [0, 0])
        if isinstance(delta, (list, tuple)) and len(delta) >= 2:
            if abs(delta[1]) >= abs(delta[0]):
                parts.append("DOWN" if delta[1] > 0 else "UP")
            else:
                parts.append("RIGHT" if delta[0] > 0 else "LEFT")
    elif action_type == "tab_focus":
        parts.append(f"tab={action.get('page_number', '?')}")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Condition metadata
# ---------------------------------------------------------------------------

def _load_condition_labels(run_dir: Path) -> Dict[str, Dict[str, str]]:
    """Load condition labels and observation modes from condition_meta.json files."""
    labels: Dict[str, Dict[str, str]] = {}
    for cond_dir in sorted(run_dir.iterdir()):
        if not cond_dir.is_dir() or cond_dir.name in ("analysis", ".git"):
            continue
        default = {"label": cond_dir.name, "observation_mode": "unknown"}
        meta_path = cond_dir / "condition_meta.json"
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                labels[cond_dir.name] = {
                    "label": meta.get("label", cond_dir.name),
                    "observation_mode": meta.get("observation_mode", "unknown"),
                }
            except Exception:
                labels[cond_dir.name] = default
        else:
            labels[cond_dir.name] = default
    return labels


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------

def _build_groups(
    episodes: List[Dict[str, Any]],
    condition_labels: Dict[str, Dict[str, str]],
) -> List[Dict[str, Any]]:
    """Group episodes by (site, condition) and compute per-group stats."""
    group_map: Dict[tuple, Dict[str, Any]] = OrderedDict()
    for ep in episodes:
        key = (ep["site"], ep["condition"])
        if key not in group_map:
            cl = condition_labels.get(ep["condition"], {})
            group_map[key] = {
                "site": ep["site"],
                "condition": ep["condition"],
                "condition_label": cl.get("label", ep["condition"]),
                "observation_mode": cl.get("observation_mode", "unknown"),
                "episodes": [],
            }
        group_map[key]["episodes"].append({
            "key": ep["key"],
            "task_id": ep["task_id"],
            "label": ep["label"],
            "intent": ep.get("intent", ""),
            "success": ep["success"],
            "score": ep["score"],
            "total_steps": ep["total_steps"],
            "steps": ep["steps"],
        })

    groups = []
    for group in group_map.values():
        total = len(group["episodes"])
        success = sum(1 for e in group["episodes"] if e.get("success") is True)
        fail = sum(1 for e in group["episodes"] if e.get("success") is False)
        group["stats"] = {
            "total": total,
            "success": success,
            "fail": fail,
            "success_rate": round(success / total, 3) if total > 0 else 0,
        }
        groups.append(group)
    return groups


# ---------------------------------------------------------------------------
# Task intents from VWA config
# ---------------------------------------------------------------------------

_VWA_CONFIG_BASE = Path(__file__).resolve().parent.parent / "external" / "visualwebarena" / "config_files" / "vwa"

def _load_task_intents() -> Dict[str, str]:
    """Load task intents from VWA config files. Returns {'{site}_task_{id}': intent}."""
    intents: Dict[str, str] = {}
    if not _VWA_CONFIG_BASE.exists():
        return intents
    for site_dir in _VWA_CONFIG_BASE.iterdir():
        if not site_dir.is_dir() or not site_dir.name.startswith("test_"):
            continue
        site = site_dir.name.replace("test_", "")
        for cfg_path in site_dir.glob("*.json"):
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                tid = cfg.get("task_id")
                intent = cfg.get("intent", "")
                if tid is not None and intent:
                    intents[f"{site}_task_{tid}"] = intent
            except Exception:
                continue
    return intents


# ---------------------------------------------------------------------------
# Episode collection
# ---------------------------------------------------------------------------

def _collect_episodes(
    run_dir: Path,
    condition_filter: Optional[str],
    task_id_filter: Optional[int],
    gallery_path: Path,
    embed: bool,
) -> List[Dict[str, Any]]:
    """Collect episodes with their steps and image sources."""
    intents = _load_task_intents()
    episodes = []
    condition_dirs = sorted(run_dir.iterdir())
    for cond_dir in condition_dirs:
        if not cond_dir.is_dir() or cond_dir.name in ("analysis", ".git"):
            continue
        if condition_filter and cond_dir.name != condition_filter:
            continue
        episodes_dir = cond_dir / "episodes"
        artifacts_dir = cond_dir / "artifacts"
        if not episodes_dir.exists():
            continue

        for jsonl_path in sorted(episodes_dir.glob("*_steps_v2.jsonl")):
            stem = jsonl_path.stem.replace("_steps_v2", "")
            parts = stem.rsplit("_task_", 1)
            if len(parts) != 2:
                continue
            site = parts[0]
            try:
                task_id = int(parts[1])
            except ValueError:
                continue
            if task_id_filter is not None and task_id != task_id_filter:
                continue

            steps = _read_steps(jsonl_path)
            if not steps:
                continue

            # Read summary
            summary_path = episodes_dir / f"{site}_task_{task_id}_summary_v2.json"
            summary = None
            if summary_path.exists():
                try:
                    with open(summary_path, "r", encoding="utf-8") as f:
                        summary = json.load(f)
                except Exception:
                    pass

            # Collect steps
            task_artifact_dir = artifacts_dir / f"{site}_task_{task_id}"
            step_data = []
            for step in steps:
                step_idx = step.get("step_idx", len(step_data))
                step_dir = task_artifact_dir / f"step_{step_idx:03d}"
                annotated = step_dir / "screenshot_annotated.png"
                raw = step_dir / "screenshot.png"
                img_path = annotated if annotated.exists() else raw

                if embed:
                    img_src = _img_to_data_uri(img_path)
                else:
                    img_src = _img_to_relative(img_path, gallery_path)

                # Extract thought from action dict (primary) or top-level (fallback)
                action = step.get("action", {})
                if isinstance(action, dict):
                    thought = str(action.get("thought", "") or "").strip()
                else:
                    thought = ""
                if not thought:
                    thought = str(step.get("thought", "") or "").strip()

                step_data.append({
                    "step_idx": step_idx,
                    "action_summary": _build_action_summary(step),
                    "thought": thought[:200],
                    "reward": step.get("reward"),
                    "img_path": img_src,
                })

            ep_key = f"{cond_dir.name}__{site}_task_{task_id}"
            label = f"{site}_task_{task_id}"
            episodes.append({
                "key": ep_key,
                "condition": cond_dir.name,
                "site": site,
                "task_id": task_id,
                "label": label,
                "intent": intents.get(label, ""),
                "steps": step_data,
                "success": summary.get("success") if summary else None,
                "score": summary.get("score") if summary else None,
                "total_steps": len(step_data),
            })

    episodes.sort(key=lambda e: (e["condition"], e["site"], e["task_id"]))
    return episodes


# ---------------------------------------------------------------------------
# HTML template (v2 — dual-view architecture)
# ---------------------------------------------------------------------------

_HTML_TEMPLATE_V2 = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Episode Gallery &mdash; {title}</title>
<style>
*{{ margin:0; padding:0; box-sizing:border-box; }}
body{{
  font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
  background:#1a1a2e; color:#e0e0e0;
}}

/* ========== Home View ========== */
#home-view{{ padding:20px 30px; max-width:1400px; margin:0 auto; }}
.home-header{{ display:flex; align-items:baseline; gap:16px; margin-bottom:18px; }}
.home-header h1{{ font-size:22px; }}
.home-header .gen-time{{ font-size:12px; color:#666; }}

.group-card{{
  margin-bottom:14px; background:#16213e; border-radius:8px;
  border:1px solid #2a2a4a; overflow:hidden;
}}
.group-header{{
  padding:12px 16px; display:flex; align-items:center; gap:12px;
  cursor:pointer; user-select:none; transition:background .15s;
}}
.group-header:hover{{ background:#1a3a5c; }}
.group-header h3{{ font-size:15px; }}
.site-badge{{
  padding:2px 8px; border-radius:4px; font-size:11px;
  font-weight:700; text-transform:uppercase;
}}
.site-shopping{{ background:#1b5e20; color:#a5d6a7; }}
.site-reddit{{ background:#bf360c; color:#ffab91; }}
.site-wikipedia{{ background:#0d47a1; color:#90caf9; }}
.site-classifieds{{ background:#4a148c; color:#ce93d8; }}

.stats-bar{{
  display:flex; align-items:center; gap:10px;
  margin-left:auto; font-size:13px; color:#aaa;
}}
.stats-bar .s{{ white-space:nowrap; }}
.stats-bar .s.ok{{ color:#4caf50; }}
.stats-bar .s.no{{ color:#f44336; }}
.progress-bar{{ width:80px; height:5px; background:#333; border-radius:3px; overflow:hidden; }}
.progress-fill{{ height:100%; background:#4caf50; border-radius:3px; }}
.group-toggle{{ font-size:12px; color:#888; min-width:14px; text-align:center; }}

.ep-table{{ display:none; width:100%; border-collapse:collapse; }}
.ep-table.expanded{{ display:table; }}
.ep-table th{{
  text-align:left; padding:5px 12px; font-size:12px; color:#666;
  border-bottom:1px solid #333; background:#0f1a30;
}}
.ep-table td{{ padding:5px 12px; font-size:13px; border-bottom:1px solid #1e1e3e; }}
.ep-table tr.ep-row{{ cursor:pointer; transition:background .1s; }}
.ep-table tr.ep-row:hover{{ background:#1a3a5c; }}

.badge{{
  display:inline-block; padding:2px 8px; border-radius:10px;
  font-size:11px; font-weight:600;
}}
.badge.success{{ background:#1b5e20; color:#a5d6a7; }}
.badge.fail{{ background:#b71c1c; color:#ef9a9a; }}
.badge.unknown{{ background:#333; color:#888; }}

/* ========== Episode View ========== */
#episode-view{{ display:none; }}

.ep-top-bar{{
  position:sticky; top:0; z-index:100;
  background:#16213e; padding:5px 16px;
  border-bottom:1px solid #333;
  display:flex; align-items:center; gap:10px;
}}
.back-btn{{
  background:none; border:1px solid #555; color:#ccc;
  padding:4px 10px; border-radius:4px; cursor:pointer; font-size:13px;
}}
.back-btn:hover{{ background:#1a3a5c; }}
.ep-title{{ font-size:15px; font-weight:600; }}
.nav-btn{{
  background:#1a3a5c; border:none; color:#ccc;
  padding:4px 10px; border-radius:4px; cursor:pointer; font-size:13px;
}}
.nav-btn:hover{{ background:#2a5a8c; }}
.nav-btn:disabled{{ opacity:.3; cursor:default; }}
.ep-spacer{{ flex:1; }}

.step-nav{{
  position:sticky; top:34px; z-index:90;
  background:#1a1a2e; padding:3px 16px;
  display:flex; gap:2px; flex-wrap:wrap;
  border-bottom:1px solid #2a2a4a;
}}
.step-dot{{
  width:22px; height:22px; border-radius:3px;
  border:1px solid #444; display:flex; align-items:center;
  justify-content:center; font-size:10px; cursor:pointer;
  transition:all .12s; color:#aaa;
}}
.step-dot:hover{{ background:#1a3a5c; border-color:#64b5f6; }}
.step-dot.active{{ background:#1a3a5c; border-color:#64b5f6; color:#fff; }}

.steps-area{{ padding:0 16px 80px; max-width:1200px; margin:0 auto; }}

.step-card{{
  margin:2px 0 40px; background:#16213e; border-radius:6px;
  overflow:hidden; border:1px solid #2a2a4a; scroll-margin-top:62px;
}}
.step-info{{
  padding:3px 12px; font-size:12px; color:#aaa;
  display:flex; gap:10px; align-items:center; background:#0f1a30;
}}
.step-info .sn{{ font-weight:700; color:#64b5f6; min-width:44px; }}
.step-info .act{{ color:#e0e0e0; font-family:monospace; font-size:11px; }}
.step-thought{{
  padding:1px 12px 2px; font-size:11px; color:#777;
  background:#0f1a30; border-top:1px solid #1a2a40; font-style:italic;
  white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
}}
.step-card img{{
  width:100%; display:block; cursor:pointer;
  object-fit:contain; background:#0a0a1a;
}}
img.zoomed{{
  position:fixed; top:0; left:0; width:100vw; height:100vh;
  object-fit:contain; z-index:1000; background:rgba(0,0,0,.92);
  cursor:zoom-out;
}}
.no-img{{ padding:40px; color:#555; text-align:center; }}
</style>
</head>
<body>

<div id="home-view"></div>
<div id="episode-view"></div>

<script type="application/json" id="gallery-data">
{data_json}
</script>

<script>
(function(){{
'use strict';

var D=JSON.parse(document.getElementById('gallery-data').textContent);
var GROUPS=D.groups, ORDER=D.episode_order, IDX=D.episode_index;
var SKEY='gallery_v2_'+D.title;

/* ---- state ---- */
var S={{view:'home',epKey:null,step:0,scrollY:0,eg:{{}}}};
function save(){{ try{{localStorage.setItem(SKEY,JSON.stringify(S));}}catch(e){{}} }}
function load(){{ try{{var s=localStorage.getItem(SKEY);if(s)Object.assign(S,JSON.parse(s));}}catch(e){{}} }}

/* ---- helpers ---- */
function ep(k){{ var l=IDX[k]; return l?GROUPS[l[0]].episodes[l[1]]:null; }}
function oi(k){{ return ORDER.indexOf(k); }}
function esc(s){{ return s?s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'):''; }}
function escA(s){{ return s?s.replace(/&/g,'&amp;').replace(/"/g,'&quot;').replace(/'/g,'&#39;'):''; }}

var $h=document.getElementById('home-view');
var $e=document.getElementById('episode-view');

/* ======== Home View ======== */
function renderHome(){{
  var h='<div class="home-header"><h1>'+esc(D.title)+'</h1>'
    +'<span class="gen-time">'+esc(D.generated_at)+'</span>'
    +'<span class="gen-time">'+ORDER.length+' episodes</span></div>';
  GROUPS.forEach(function(g,gi){{
    var sr=(g.stats.success_rate*100).toFixed(1);
    var sc='site-'+g.site;
    var ex=S.eg[gi];
    h+='<div class="group-card">'
      +'<div class="group-header" data-gi="'+gi+'">'
      +'<span class="site-badge '+sc+'">'+esc(g.site)+'</span>'
      +'<h3>'+esc(g.condition_label)+'</h3>'
      +'<span style="color:#666;font-size:12px">'+esc(g.observation_mode)+'</span>'
      +'<div class="stats-bar">'
      +'<span class="s">'+g.stats.total+'</span>'
      +'<span class="s ok">'+g.stats.success+' pass</span>'
      +'<span class="s no">'+g.stats.fail+' fail</span>'
      +'<div class="progress-bar"><div class="progress-fill" style="width:'+sr+'%"></div></div>'
      +'<span class="s">'+sr+'%</span>'
      +'</div>'
      +'<span class="group-toggle">'+(ex?'&#9660;':'&#9654;')+'</span>'
      +'</div>';
    h+='<table class="ep-table'+(ex?' expanded':'')+'" data-gi="'+gi+'">'
      +'<thead><tr><th>Task</th><th>Status</th><th>Steps</th><th>Score</th></tr></thead><tbody>';
    g.episodes.forEach(function(e){{
      var c=e.success===true?'success':e.success===false?'fail':'unknown';
      var sl=e.success===true?'PASS':e.success===false?'FAIL':'&mdash;';
      var sc2=e.score!=null?e.score.toFixed(2):'&mdash;';
      h+='<tr class="ep-row" data-key="'+escA(e.key)+'">'
        +'<td>'+esc(e.label)+'</td>'
        +'<td><span class="badge '+c+'">'+sl+'</span></td>'
        +'<td>'+e.total_steps+'</td>'
        +'<td>'+sc2+'</td></tr>';
    }});
    h+='</tbody></table></div>';
  }});
  $h.innerHTML=h;

  /* bind group toggles */
  $h.querySelectorAll('.group-header').forEach(function(hdr){{
    hdr.addEventListener('click',function(){{
      var gi=parseInt(hdr.dataset.gi);
      var t=$h.querySelector('.ep-table[data-gi="'+gi+'"]');
      var tg=hdr.querySelector('.group-toggle');
      var x=t.classList.toggle('expanded');
      tg.innerHTML=x?'&#9660;':'&#9654;';
      S.eg[gi]=x||undefined;
      if(!x) delete S.eg[gi];
      save();
    }});
  }});
  /* bind episode rows */
  $h.querySelectorAll('.ep-row').forEach(function(r){{
    r.addEventListener('click',function(){{ goEp(r.dataset.key,0); }});
  }});
}}

/* ======== Episode View ======== */
function renderEp(k){{
  var e=ep(k); if(!e) return;
  var o=oi(k), hp=o>0, hn=o<ORDER.length-1;
  var c=e.success===true?'success':e.success===false?'fail':'unknown';
  var sl=e.success===true?'PASS':e.success===false?'FAIL':'&mdash;';
  var h='<div class="ep-top-bar">'
    +'<button class="back-btn" id="eb">&#8592; Home</button>'
    +'<span class="ep-title">'+esc(e.label)+'</span>'
    +'<span class="badge '+c+'">'+sl+'</span>';
  if(e.score!=null) h+='<span style="color:#888;font-size:12px">score='+e.score.toFixed(2)+'</span>';
  if(e.intent) h+='<span style="color:#aaa;font-size:12px;max-width:500px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="'+escA(e.intent)+'">'+esc(e.intent)+'</span>';
  h+='<span class="ep-spacer"></span>'
    +'<button class="nav-btn" id="enp"'+(hp?'':' disabled')+'>&#8592; Prev</button>'
    +'<span style="color:#666;font-size:12px">'+(o+1)+'/'+ORDER.length+'</span>'
    +'<button class="nav-btn" id="enn"'+(hn?'':' disabled')+'>Next &#8594;</button>'
    +'</div>';

  /* step dots */
  var si=Math.min(S.step, e.steps.length-1);
  if(si<0) si=0;
  S.step=si;
  h+='<div class="step-nav">';
  e.steps.forEach(function(s,i){{
    h+='<div class="step-dot'+(i===si?' active':'')+'" data-si="'+i+'">'+i+'</div>';
  }});
  h+='</div>';

  /* step cards */
  h+='<div class="steps-area">';
  e.steps.forEach(function(s,i){{
    h+='<div class="step-card" id="sc'+i+'">'
      +'<div class="step-info">'
      +'<span class="sn">Step '+s.step_idx+'</span>'
      +'<span class="act">'+esc(s.action_summary)+'</span>';
    if(s.reward!=null)
      h+='<span style="margin-left:auto;color:'+(s.reward>0?'#4caf50':'#888')+';font-size:12px">r='+s.reward+'</span>';
    h+='</div>';
    if(s.thought)
      h+='<div class="step-thought">'+esc(s.thought)+'</div>';
    if(s.img_path)
      h+='<img src="'+escA(s.img_path)+'">';
    else
      h+='<div class="no-img">No screenshot</div>';
    h+='</div>';
  }});
  h+='</div>';
  $e.innerHTML=h;

  /* bind */
  document.getElementById('eb').addEventListener('click',goHome);
  if(hp) document.getElementById('enp').addEventListener('click',function(){{ goEp(ORDER[o-1],0); }});
  if(hn) document.getElementById('enn').addEventListener('click',function(){{ goEp(ORDER[o+1],0); }});
  $e.querySelectorAll('.step-dot').forEach(function(d){{
    d.addEventListener('click',function(){{ scrollStep(parseInt(d.dataset.si)); }});
  }});
}}

/* ---- view switching ---- */
function goHome(){{
  $e.style.display='none'; $h.style.display='block';
  S.view='home'; S.epKey=null; save();
  window.scrollTo(0,S.scrollY||0);
}}
function goEp(k,si){{
  if(!ep(k)) return;
  S.scrollY=window.scrollY;
  S.view='episode'; S.epKey=k; S.step=si||0; save();
  try{{
    renderEp(k);
    $h.style.display='none'; $e.style.display='block';
    if(si>0) requestAnimationFrame(function(){{ scrollStep(si); }});
    else window.scrollTo(0,0);
    preloadAdjacent(k);
  }}catch(err){{
    $e.innerHTML='<div style="padding:40px;color:#f44"><h2>Render error</h2><pre>'+esc(err.message+'\\n'+err.stack)+'</pre></div>';
    $e.style.display='block'; $h.style.display='none';
    console.error('goEp error:',err);
  }}
}}
/* preload images of prev/next episodes to avoid flicker on switch */
var _preloaded={{}};
function preloadAdjacent(k){{
  var o=oi(k);
  var keys=[];
  if(o>0) keys.push(ORDER[o-1]);
  if(o<ORDER.length-1) keys.push(ORDER[o+1]);
  keys.forEach(function(pk){{
    if(_preloaded[pk]) return;
    _preloaded[pk]=true;
    var e=ep(pk); if(!e) return;
    e.steps.forEach(function(s){{
      if(s.img_path){{ var img=new Image(); img.src=s.img_path; }}
    }});
  }});
}}

function scrollStep(i){{
  S.step=i; save();
  var c=document.getElementById('sc'+i);
  if(c) c.scrollIntoView({{behavior:'smooth',block:'start'}});
  $e.querySelectorAll('.step-dot').forEach(function(d){{
    d.classList.toggle('active',parseInt(d.dataset.si)===i);
  }});
}}

/* ---- image zoom ---- */
document.addEventListener('click',function(ev){{
  if(ev.target.tagName==='IMG'&&ev.target.closest('.step-card'))
    ev.target.classList.toggle('zoomed');
}});

/* ---- keyboard ---- */
document.addEventListener('keydown',function(ev){{
  if(ev.target.tagName==='INPUT'||ev.target.tagName==='TEXTAREA') return;
  if(S.view==='episode'){{
    var e=ep(S.epKey); if(!e) return;
    var o=oi(S.epKey);
    if(ev.key==='Escape'){{
      var z=document.querySelector('.zoomed');
      if(z) z.classList.remove('zoomed'); else goHome();
      ev.preventDefault();
    }} else if(ev.key==='ArrowLeft'||ev.key==='k'){{
      if(o>0) goEp(ORDER[o-1],0); ev.preventDefault();
    }} else if(ev.key==='ArrowRight'||ev.key==='j'){{
      if(o<ORDER.length-1) goEp(ORDER[o+1],0); ev.preventDefault();
    }} else if(ev.key==='ArrowUp'){{
      if(S.step>0) scrollStep(S.step-1); ev.preventDefault();
    }} else if(ev.key==='ArrowDown'){{
      if(S.step<e.steps.length-1) scrollStep(S.step+1); ev.preventDefault();
    }}
  }}
}});

/* ---- scroll sync (episode view) ---- */
var sT=null;
window.addEventListener('scroll',function(){{
  if(S.view!=='episode') return;
  clearTimeout(sT);
  sT=setTimeout(function(){{
    var e=ep(S.epKey); if(!e) return;
    var vt=window.scrollY+120;
    for(var i=e.steps.length-1;i>=0;i--){{
      var c=document.getElementById('sc'+i);
      if(c&&c.offsetTop<=vt){{
        S.step=i;
        $e.querySelectorAll('.step-dot').forEach(function(d){{
          d.classList.toggle('active',parseInt(d.dataset.si)===i);
        }});
        save(); break;
      }}
    }}
  }},150);
}});

/* ---- auto-refresh ---- */
setInterval(function(){{ save(); location.reload(); }},60000);

/* ---- init ---- */
load();
renderHome();
if(S.view==='episode'&&S.epKey&&ep(S.epKey)){{
  goEp(S.epKey,S.step);
}} else {{
  S.view='home'; $h.style.display='block'; $e.style.display='none';
  if(S.scrollY) requestAnimationFrame(function(){{ window.scrollTo(0,S.scrollY); }});
}}

}})();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def generate_gallery(
    run_dir: Path,
    condition: Optional[str],
    task_id: Optional[int],
    embed: bool,
) -> Path:
    gallery_path = run_dir / "gallery.html"
    episodes = _collect_episodes(run_dir, condition, task_id, gallery_path, embed)
    if not episodes:
        print("No episodes found.")
        raise SystemExit(1)

    condition_labels = _load_condition_labels(run_dir)
    groups = _build_groups(episodes, condition_labels)

    # Build global ordering and O(1) index
    episode_order: List[str] = []
    episode_index: Dict[str, List[int]] = {}
    for gi, group in enumerate(groups):
        for ei, ep in enumerate(group["episodes"]):
            episode_order.append(ep["key"])
            episode_index[ep["key"]] = [gi, ei]

    title = run_dir.name
    if condition:
        title += f" / {condition}"
    if task_id is not None:
        title += f" / task_{task_id}"

    data = {
        "title": title,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "groups": groups,
        "episode_order": episode_order,
        "episode_index": episode_index,
    }

    data_json = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    # Prevent </script> in JSON from closing the script tag
    data_json = data_json.replace("</", "<\\/")

    html_content = _HTML_TEMPLATE_V2.format(
        title=html_mod.escape(title),
        data_json=data_json,
    )

    gallery_path.write_text(html_content, encoding="utf-8")
    print(f"Gallery: {gallery_path}  ({len(episodes)} episodes)")
    return gallery_path


def main():
    parser = argparse.ArgumentParser(description="Generate screenshot gallery HTML")
    parser.add_argument("--run-dir", required=True, help="Run directory")
    parser.add_argument("--condition", default=None, help="Filter to condition_id")
    parser.add_argument("--task-id", type=int, default=None, help="Filter to task_id")
    parser.add_argument(
        "--embed", action="store_true",
        help="Embed images as base64 (larger file but self-contained)",
    )
    args = parser.parse_args()
    generate_gallery(Path(args.run_dir), args.condition, args.task_id, args.embed)


if __name__ == "__main__":
    main()
