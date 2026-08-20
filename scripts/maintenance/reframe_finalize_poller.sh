#!/usr/bin/env bash
# reframe_finalize_poller.sh — DGX-side unattended finalisation for the reframe chain.
#
# Runs on the DGX, not the A100, for one reason: the A100 has no git credentials
# (`user.name` and `credential.helper` both unset, read-only remote access), so it
# cannot register anything. Registration edits a tracked file and must be committed.
# Data also only moves DGX-ward: the A100 cannot reach the DGX, so the sync has to be
# pulled from this side.
#
# Each pass does three things, all idempotent:
#   1. `sync_a100_results.sh` — pull whatever has landed
#   2. for each declared pair whose BOTH arms are now complete: register it into
#      CLEAN_PAIRS via register_replicate_pair.py (which verifies and rolls back
#      on its own if the edit would break `literal_eval` — see §469.5)
#   3. commit the registration
#
# IT PUSHES — but only these commits. The standing rule is that pushing needs explicit
# confirmation; user granted it for this loop specifically (2026-08-19, in answer to
# "要不要我把 push 也纳入自动化"). The grant is scoped by what this script can even
# produce: a commit touching CLEAN_PAIRS and the two regenerated inventory artefacts,
# for a pair that was declared before the fire. It is not a general licence, and
# nothing else in the repo should read it as one.
#
# ⚠️ B-1982 (2026-08-20) — THE ABOVE PARAGRAPH WAS FALSE AS WRITTEN, AND HAD TO BE MADE
# TRUE. `git push` pushes the BRANCH, not the commit the script just made. So the grant's
# premise ("scoped by what this script can even produce") did not hold: any commit any
# human left on the branch went out with it. Observed 2026-08-20 09:23Z — the retry block
# at the bottom of the loop pushed three unrelated commits made 11 minutes earlier in an
# interactive session, none of them a registration, without anyone confirming. Both push
# sites now go through `_scoped_push`, which refuses unless EVERY commit ahead of origin
# is one this script authored (subject line + touched-paths allowlist).
#
# A failed push is reported, never swallowed: an unpushed registration looks identical
# to a registration that never happened when the next session looks at the remote.
#
# The pairs it watches are the ones DECLARED in
# docs/checkpoints/pre_run/reframe_chain_launch_intent_20260819.md. Nothing is
# discovered dynamically: a pair that was not declared before the fire does not get
# registered by a background job (§469.7 — the whole point is that registration is
# not a choice made after seeing the number).
set -uo pipefail
REPO="${P79_REPO:-/home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents}"
cd "$REPO" || exit 1
NTFY="${NTFY_TOPIC:-p79-exp-dgx-spark}"
PY="${REPO}/.venv/bin/python3"
TS="$(date -u +%Y%m%d_%H%M%S)"
LOG="${REPO}/logs/reframe_finalize_${TS}.log"
DEADLINE_UTC="${DEADLINE_UTC:-2026-09-08}"
INTERVAL="${INTERVAL:-1800}"

say()  { echo "[finalize $(date -u '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
push() { curl -s -m 20 -H "Title: $1" -d "$2" "https://ntfy.sh/${NTFY}" >/dev/null 2>&1 || true; }

# B-1982. Push only if every commit ahead of origin is one this script produced.
# Two independent conditions, because either alone is forgeable by accident: the
# subject line this script writes, AND the three paths it is allowed to touch.
_POLLER_SUBJECT_RE='^注册 .* 进 CLEAN_PAIRS \(reframe chain 自动收尾\)$'
_poller_authored_only() {
  local br ahead sha files f
  br="$(git rev-parse --abbrev-ref HEAD)" || return 1
  ahead="$(git log --format=%H "origin/${br}..HEAD" 2>/dev/null)" || return 1
  [ -n "$ahead" ] || return 1                      # nothing to push
  while read -r sha; do
    [ -z "$sha" ] && continue
    git log -1 --format=%s "$sha" | grep -qE "$_POLLER_SUBJECT_RE" || return 1
    files="$(git show --name-only --format= "$sha")"
    while read -r f; do
      [ -z "$f" ] && continue
      case "$f" in
        scripts/analysis/aggregate_noise_floor_inventory.py) ;;
        docs/analysis/cross_sites/noise_floor_inventory.md)  ;;
        docs/analysis/cross_sites/noise_floor_inventory.json) ;;
        *) return 1 ;;
      esac
    done <<< "$files"
  done <<< "$ahead"
  return 0
}
_scoped_push() {  # returns 0 on a successful push, 1 otherwise; never pushes foreign commits
  local br
  br="$(git rev-parse --abbrev-ref HEAD)"
  if ! _poller_authored_only; then
    local blocked; blocked="$(git log --oneline "origin/${br}..HEAD" 2>/dev/null)"
    if [ -n "$blocked" ]; then
      say "REFUSING to push: commits ahead of origin/${br} were not authored by this script (B-1982)"
      echo "$blocked" | while read -r l; do say "    $l"; done
      # B-1989 (2026-08-20): notify on CHANGE, not on every pass. The first version
      # pushed one ntfy per 30-minute pass for as long as the branch stayed ahead —
      # 5 identical messages in 2 hours, on track for ~48/day, none carrying anything
      # the previous one had not. A notification that repeats without new information
      # is how a channel stops being read, and this channel is also where the fire's
      # real alarms land. Fingerprint what is blocked; speak only when it changes.
      local fp seen_fp="" fpfile="${REPO}/logs/.poller_push_block.fp"
      fp="$(printf '%s' "$blocked" | cksum | awk '{print $1}')"
      [ -f "$fpfile" ] && seen_fp="$(cat "$fpfile" 2>/dev/null)"
      if [ "$fp" != "$seen_fp" ]; then
        printf '%s' "$fp" > "$fpfile"
        local n top
        n="$(printf '%s\n' "$blocked" | grep -c .)"
        top="$(printf '%s\n' "$blocked" | head -1)"
        push "reframe poller 拒绝 push (${n} 个 commit)" \
          "分支上有 ${n} 个非本脚本的 commit, 按 B-1982 不代推, 需要人工 push。最新: ${top}"
      fi
    fi
    return 1
  fi
  if git push -q 2>>"$LOG"; then
    rm -f "${REPO}/logs/.poller_push_block.fp"   # next distinct block should speak again
    return 0
  fi
  return 1
}

# label | canonical glob | replicate glob | cond id | expected n
# A1 is whichever B5 dom run is OLDER, A2 the newer — assigned by fire order, not by
# which number is nicer. Resolved at match time so the declaration needs no run ids.
DECLARED_PAIRS="B5.cls.dom|B5_dom_classifieds_2*|phase1_dom_router_0|224"

_complete() {  # <cond dir, repo-relative>
  local f="${REPO}/$1/condition_summary_v2.json"
  [ -s "$f" ] || return 1
  SUMMARY_PATH="$f" EXPECT_N="$2" "$PY" -c "
import json, os, sys
d = json.load(open(os.environ['SUMMARY_PATH']))
ep = d.get('episodes', d.get('total_tasks', d.get('num_tasks', d.get('scored_task_count', 0))))
sys.exit(0 if isinstance(ep, int) and ep == int(os.environ['EXPECT_N']) else 1)
" 2>/dev/null
}

say "poller armed — interval ${INTERVAL}s, deadline ${DEADLINE_UTC}, repo ${REPO}"

while [[ "$(date -u +%F)" < "$DEADLINE_UTC" ]]; do
  # ---- 1. pull whatever landed -------------------------------------------
  # SKIP_SYNC exists so the registration half can be exercised on its own — both for
  # testing and for the case where sync is broken but data is already local.
  if [ "${SKIP_SYNC:-0}" = "1" ]; then
    say "sync skipped (SKIP_SYNC=1)"
  elif bash scripts/maintenance/sync_a100_results.sh >>"$LOG" 2>&1; then
    say "sync ok"
  else
    say "sync returned nonzero (continuing; next pass retries)"
  fi

  # ---- 2. register any declared pair that is now complete on both arms ----
  while IFS='|' read -r label glob cond expn; do
    [ -z "$label" ] && continue
    "$PY" scripts/analysis/register_replicate_pair.py --label "$label" \
        --canonical x --replicate y --expected-n "$expn" 2>/dev/null \
        | grep -q "already registered" && continue

    # two runs matching the glob = the pair; oldest is arm A by fire order
    mapfile -t runs < <(ls -dtr "${REPO}"/results/visualwebarena/phase1/${glob} 2>/dev/null)
    if [ "${#runs[@]}" -lt 2 ]; then
      say "  ${label}: ${#runs[@]}/2 runs present — waiting"
      continue
    fi
    A="results/visualwebarena/phase1/$(basename "${runs[0]}")/${cond}"
    B="results/visualwebarena/phase1/$(basename "${runs[-1]}")/${cond}"
    if ! _complete "$A" "$expn" || ! _complete "$B" "$expn"; then
      say "  ${label}: both runs present but not both complete — waiting"
      continue
    fi
    say "  ${label}: both arms complete → registering"
    if "$PY" scripts/analysis/register_replicate_pair.py \
         --label "$label" --canonical "$A" --replicate "$B" --expected-n "$expn" \
         --note "reframe chain, intent reframe_chain_launch_intent_20260819.md" >>"$LOG" 2>&1; then
      git add scripts/analysis/aggregate_noise_floor_inventory.py \
              docs/analysis/cross_sites/noise_floor_inventory.md \
              docs/analysis/cross_sites/noise_floor_inventory.json 2>/dev/null
      git commit -q -m "注册 ${label} 进 CLEAN_PAIRS (reframe chain 自动收尾)

发车前意图已在 reframe_chain_launch_intent_20260819.md 声明该 pair,
本次为 '落地后照单全收' 的执行 (§469.7)。两臂各 ${expn} episode 完整,
register_replicate_pair.py 已验证 literal_eval 仍可用且 validator 认得
新条目 — 那条链断了会把所有 replicate 判成 ghost (§469.5)。

未 push (push 需显式确认)。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>" >>"$LOG" 2>&1
      # push only the branch we are on, and only if the commit above actually made one
      if _scoped_push; then
        say "  ${label}: registered + committed + pushed"
        push "reframe: ${label} 已注册" "两臂完整, 已写入 CLEAN_PAIRS, commit + push 完成。"
      else
        say "  ${label}: registered + committed but PUSH FAILED — see ${LOG}"
        push "reframe: ${label} 注册了但 push 失败" \
             "本地 commit 在, 远端没有。下次 pass 会重试; 若持续失败请手动 push。"
      fi
    else
      say "  ${label}: registration REFUSED or rolled back — see ${LOG}"
      push "reframe 注册失败" "${label} 注册被拒或已回滚, 查 ${LOG}"
    fi
  done <<< "$DECLARED_PAIRS"

  # Retry a push that failed on an earlier pass. Without this, one transient network
  # failure would leave the registration local forever while the ntfy that reported it
  # scrolls away.
  if [ -n "$(git log --oneline "origin/$(git rev-parse --abbrev-ref HEAD)..HEAD" 2>/dev/null)" ]; then
    if _scoped_push; then
      say "pushed commits that an earlier pass could not"
    fi
  fi

  sleep "$INTERVAL"
done

say "poller reached deadline ${DEADLINE_UTC} — exiting"
push "reframe poller 退出" "到达 deadline ${DEADLINE_UTC}"
