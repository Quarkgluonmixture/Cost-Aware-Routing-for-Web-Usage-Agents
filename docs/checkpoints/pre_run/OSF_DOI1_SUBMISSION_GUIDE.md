---
type: osf-doi1-submission-guide
status: ready-for-user-action
generated_at: 2026-05-18T21:34Z
witness_tag: preregistration-doi1-witnessed-20260518T211628Z
witness_commit: 5edac3b
substance_lock_tag: preregistration-locked-q3a
substance_lock_commit: 72b93c9
deposit_dir: docs/checkpoints/pre_run/osf_deposit_DOI1_20260518T211628Z/
fire_status: Fire-3 attempt #6 LIVE since 2026-05-18T21:27:28Z (cls chain running, ~1-2 wk wallclock)
---

# OSF DOI 1 Submission — Step-by-Step User Guide

> **Status**: ready for user manual action. **All technical prep complete**; submission requires user manual sign-off + OSF UI navigation. Fire-3 is independently running in background — submission can happen NOW (Fire-3 doesn't block submission and submission doesn't block Fire-3).

## TL;DR

```
1. Fill 2 manual sign-off rows in release_redaction_checklist.md
2. git commit + git push the sign-off
3. Browse OSF UI: https://osf.io/registries/
4. Create new registration linked to tag preregistration-doi1-witnessed-20260518T211628Z
5. Upload bundle osf_deposit_DOI1_20260518T211628Z/ (30 files)
6. Click Submit
7. Post-DOI: backfill osf_lock_manifest.md §2 with DOI string
```

Time estimate: **15-30 min user action** for steps 1-6; DOI assignment 0-48h auto-approval post-submission.

---

## Step 1 — Sign release_redaction sign-off rows (REQUIRED before submission)

Open `docs/checkpoints/pre_run/release_redaction_checklist.md` in your editor. Find the section `## Sign-off log` → `### Pending author manual sign-offs`.

You have 2 rows to fill:

### Row A: Author manual pre-submission review

Before signing, do these 4 manual checks (per the document's "Manual review steps" section):

```bash
# (a) Browse paper drafts for TODO_REDACT markers
grep -rn 'TODO_REDACT' docs/checkpoints/paper_drafts/

# (b) Diff deposit tarball vs working tree (look for accidental private files)
diff -r docs/checkpoints/pre_run/osf_deposit_DOI1_20260518T211628Z/ docs/checkpoints/pre_run/ | grep -E '\.env|\.auth|api_key|password|credential'

# (c) Cold-clone smoke (verify deposit works standalone)
cd /tmp && rm -rf p79-doi1-coldclone && \
  cp -r /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents/docs/checkpoints/pre_run/osf_deposit_DOI1_20260518T211628Z p79-doi1-coldclone && \
  cd p79-doi1-coldclone && sha256sum -c MANIFEST_SHA256.txt | tail -5

# (d) Random episode JSON spot-check (verify no PII / API tokens)
# (since Fire-3 just started, only 1 partial run_dir; episode_summary_v2 may be 0
# at submission moment which IS the intended pre-canonical-outcome state per
# witness file artifact_existence_check_doi1_canonical_20260518T211628Z.txt)
```

Then edit the row in release_redaction_checklist.md, replacing `<to be filled at author sign-off>` with today's date + `PASS` + any minor notes:

```markdown
| 2026-05-18T<HH:MM>Z | Author (Jiaming) — manual pre-submission review | PASS | (a) no TODO_REDACT in paper_drafts ✓ (b) no .env / .auth / api_key / credential in deposit ✓ (c) MANIFEST_SHA256 all match cold-clone ✓ (d) episode_summary_v2 count = 0 at deposit (pre-canonical-outcome anchor) ✓ |
```

### Row B: Pre-OSF-deposit final signature

Done right before clicking OSF Submit. Verify:

```bash
# (a) Deposit dir matches MANIFEST_SHA256.txt byte-for-byte
cd docs/checkpoints/pre_run/osf_deposit_DOI1_20260518T211628Z/
sha256sum -c MANIFEST_SHA256.txt | grep -v ': OK$'  # output should be empty

# (b) All TBD fields filled (only osf_doi / osf_guid / osf_submitted_at_utc should remain TBD-ALLOW)
grep -E '<to-be|<TBD' pre_run/DOI_1_README.md | grep -v 'TBD-ALLOW'  # should be empty

# (c) Anchor SHAs/tags cross-cited consistently
grep -c '72b93c9\|preregistration-locked-q3a' pre_run/DOI_1_README.md  # should be ≥ 2
grep -c '5edac3b\|preregistration-doi1-witnessed-20260518T211628Z' pre_run/DOI_1_README.md  # should be ≥ 1

# (d) NO legacy anchor cited as "current"
grep -E 'current.*ef609a3\|current.*88521b9e\|current.*preregistration-locked[^-]' pre_run/DOI_1_README.md  # should be empty
```

Then edit row B:

```markdown
| 2026-05-18T<HH:MM>Z | Pre-OSF-deposit gate (Author final signature) | PASS | (a) MANIFEST_SHA256 verify clean ✓ (b) no un-allowlisted TBD ✓ (c) canonical anchor cross-cited ≥2 times ✓ (d) no legacy anchor mis-cited as current ✓ |
```

## Step 2 — Commit + push the sign-off

```bash
cd /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
git add docs/checkpoints/pre_run/release_redaction_checklist.md
git commit -m "sign-off(OSF DOI 1 submission): author manual + pre-deposit gates PASS — ready for OSF UI submit

Both manual sign-off rows filled per release_redaction_checklist.md
'Pending author manual sign-offs' section. 4-item manual review +
4-item pre-deposit byte-level verification both completed.

Next: OSF UI submission of bundle osf_deposit_DOI1_20260518T211628Z/
at https://osf.io/registries/ linked to GitHub tag
preregistration-doi1-witnessed-20260518T211628Z (commit 5edac3b)."
git push origin master
```

Also sync the sign-off into the deposit copy + recompute MANIFEST_SHA256:

```bash
cp docs/checkpoints/pre_run/release_redaction_checklist.md \
   docs/checkpoints/pre_run/osf_deposit_DOI1_20260518T211628Z/pre_run/release_redaction_checklist.md

# Recompute MANIFEST
cd docs/checkpoints/pre_run/osf_deposit_DOI1_20260518T211628Z/
find . -type f \( -name "*.md" -o -name "*.txt" -o -name "*.bib" \) ! -name "MANIFEST_SHA256.txt" | sort | nl | while read n p; do
  H=$(sha256sum "$p" | cut -d' ' -f1)
  printf '%s\t%s  %s\n' "$n" "$H" "$p"
done > MANIFEST_SHA256.txt

cd /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
git add docs/checkpoints/pre_run/osf_deposit_DOI1_20260518T211628Z/
git commit -m "sync(OSF DOI 1 deposit): sign-off + MANIFEST recompute post author sign-off"
git push origin master
```

## Step 3 — Navigate to OSF UI

URL: **https://osf.io/registries/**

Make sure you're logged in to OSF with your account. If you don't have an OSF account yet:
- Sign up at https://osf.io/register/ (free, ORCID linkage recommended)
- Verify email

## Step 4 — Create new pre-registration

On the OSF Registries page:

1. Click **"Add new registration"** button (top-right)
2. Select registration provider: **"OSF Registries"** (the default open framework, NOT a discipline-specific one — paper §3 doesn't claim a specific OSF-provider registration schema)
3. Select schema: **"OSF Standard Pre-Data Collection Registration"** (or equivalent; the doctrine-aligned one)
4. Title (paste exactly):
   ```
   Phantom routing space: cost-aware deployment of phantom-mode observation in web usage agents (Phase 1a pre-registration)
   ```
5. Description (paste; multi-line OK):
   ```
   Pre-registration for Phase 1a (cls + red × B0/B1/B2 × 6 observation modes
   + 1 learned router per cell, 42 conditions / 6 statistical cells) of the
   phantom-SoM paper. Gating hypotheses H1 (deployment hero), H2 (drop-in
   property), H3 (structural axis evidence) governed by fixed-effects
   inverse-variance pooling over the 6 planned (site, model) cells per
   decision "3A" 2026-05-14 finite-population estimand. H10 paper §6
   router section-level gate (per-cell paired-bootstrap 95% Pareto
   non-dominance + fixed-cell 5/6 grid-level operational criterion) is
   decoupled from R1-R5 framing per Q3=A doctrine alignment 2026-05-18.

   This DOI 1 captures the pre-canonical-outcome-creation cryptographic
   witness at Fire-3 launch (tier 1 strictest, all canonical-pattern
   counts = 0 at capture moment). Reproducibility bundle (Pass-1 + Pass-2
   data + analysis + final paper drafts) will be deposited separately
   as DOI 2 with explicit `cited_by DOI 1` ~2-3 weeks post-fire.

   Git anchor: tag preregistration-doi1-witnessed-20260518T211628Z at
   commit 5edac3b on master; substance-lock tag preregistration-locked-q3a
   at commit 72b93c9. Submission timestamp on this OSF page is the
   cryptographic pre-canonical-outcome-creation anchor.
   ```

## Step 5 — Link GitHub repository + tag

OSF supports "add-on" connection to GitHub. To link the registration to your locked code state:

1. In the registration draft, look for **"Linked Resources"** or **"Add-ons"** section
2. Connect GitHub: **https://github.com/Quarkgluonmixture/Cost-Aware-Routing-for-Web-Usage-Agents**
3. Specify the tag reference URL:
   ```
   https://github.com/Quarkgluonmixture/Cost-Aware-Routing-for-Web-Usage-Agents/tree/preregistration-doi1-witnessed-20260518T211628Z
   ```
4. Verify: clicking the URL should show repo state at commit 5edac3b with the deposit bundle visible at `docs/checkpoints/pre_run/osf_deposit_DOI1_20260518T211628Z/`

Some OSF templates ask for a "Project URL" — paste the tag URL above. Others let you upload separately (Step 6).

## Step 6 — Upload bundle

**Option A (preferred if OSF supports git anchor)**: Just cite the GitHub tag URL. OSF will fetch + freeze the tree at that commit. No separate upload needed.

**Option B (if OSF requires bundle upload)**: Upload the 30 files:

```
osf_deposit_DOI1_20260518T211628Z/
├── MANIFEST_SHA256.txt
├── paper_drafts/   (9 files: section1-8 + paper.bib)
└── pre_run/        (20 files including DOI_1_README, preregistration, osf_lock_manifest, etc.)
    └── retracted/  (1 file: VOID Fire-2 witness for forensic transparency)
```

Bundle is at:
```
/home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents/docs/checkpoints/pre_run/osf_deposit_DOI1_20260518T211628Z/
```

Use OSF's drag-and-drop UI. Total bundle size ~800KB so upload is fast.

**Verification after upload**: download the OSF-stored bundle + run `sha256sum -c MANIFEST_SHA256.txt` to confirm all 29 file hashes match.

## Step 7 — Click Submit (the cryptographic anchor moment)

⏰ **THIS IS THE CRITICAL MOMENT**.

The OSF submission timestamp is recorded as immutable metadata on the registration. This timestamp = the cryptographic external anchor that proves pre-canonical-outcome-creation registration.

Verify before clicking:
- [ ] Title + description look right
- [ ] GitHub tag URL linked (or bundle uploaded)
- [ ] No typos in metadata

Click **"Submit"** (you may need to confirm in a modal — read the OSF terms about registration immutability before confirming).

The OSF page should display:
- Submission timestamp UTC (record this immediately!)
- Registration status: "Pending admin approval" (auto-approve default)
- GUID assigned (URL fragment `osf.io/xxxxx`)

## Step 8 — Wait for DOI assignment (0-48h auto-approval)

Per help.osf.io/article/330, OSF Standard Registries auto-approve within 48 hours unless flagged by moderation. You can also manually request earlier review if needed.

When approved:
- DOI string assigned (format `10.17605/OSF.IO/XXXXX`)
- Registration becomes publicly viewable + permanently citable

## Step 9 — Post-DOI backfill

Once DOI is assigned, fill the TBD-ALLOW frontmatter fields in DOI_1_README + osf_lock_manifest + (live + deposit copies):

```bash
cd /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents

# In docs/checkpoints/pre_run/DOI_1_README.md frontmatter, replace these:
#   osf_doi: <to-be-assigned post-OSF-public-registration-approval>
#   osf_guid: <to-be-assigned at submission>
#   osf_submitted_at_utc: <to-be-recorded at submission>
# with the actual values from OSF UI.

# Then sync to deposit copy
cp docs/checkpoints/pre_run/DOI_1_README.md \
   docs/checkpoints/pre_run/osf_deposit_DOI1_20260518T211628Z/pre_run/DOI_1_README.md

# In docs/checkpoints/pre_run/osf_lock_manifest.md §2 witness chain table:
# fill the OSF DOI 1 row's <pending DOI 1 assignment> → actual DOI string

# Sync osf_lock_manifest to deposit too:
cp docs/checkpoints/pre_run/osf_lock_manifest.md \
   docs/checkpoints/pre_run/osf_deposit_DOI1_20260518T211628Z/pre_run/osf_lock_manifest.md

# Recompute MANIFEST_SHA256
cd docs/checkpoints/pre_run/osf_deposit_DOI1_20260518T211628Z/
find . -type f \( -name "*.md" -o -name "*.txt" -o -name "*.bib" \) ! -name "MANIFEST_SHA256.txt" | sort | nl | while read n p; do
  H=$(sha256sum "$p" | cut -d' ' -f1)
  printf '%s\t%s  %s\n' "$n" "$H" "$p"
done > MANIFEST_SHA256.txt

cd /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents

# Also promote git tag preregistration-doi1-witnessed → preregistration-doi1-minted-<DOI>:
git tag -a preregistration-doi1-minted-osf-<DOI-suffix> 5edac3b \
  -m "OSF DOI 1 minted at OSF GUID <GUID> + DOI 10.17605/OSF.IO/<XXXXX>; submission timestamp <UTC>; witness file artifact_existence_check_doi1_canonical_20260518T211628Z.txt SHA-256 011fa4c07d76798a57070d9aea8b1653ca50dfe432b9c4dee689b96ee9cc6691"

git add docs/checkpoints/pre_run/
git commit -m "post(OSF DOI 1 minted): backfill TBD fields with DOI + GUID + submission UTC

DOI: 10.17605/OSF.IO/<XXXXX>
GUID: osf.io/<XXXXX>
Submitted UTC: <YYYY-MM-DDTHH:MM:SSZ>
OSF admin approved at: <UTC>

Cite in paper as:
  OSF preregistration 10.17605/OSF.IO/<XXXXX>, submitted <UTC>,
  pre-canonical-outcome-creation witness for Fire-3 (empirical zero-state
  check per artifact_existence_check_doi1_canonical_20260518T211628Z.txt
  SHA-256 011fa4c07d76798a57070d9aea8b1653ca50dfe432b9c4dee689b96ee9cc6691);
  registered Git tag preregistration-doi1-minted-osf-<XXXXX> at SHA 5edac3b."
git push origin master
git push origin preregistration-doi1-minted-osf-<XXXXX>
```

## Error Recovery

### If OSF upload fails mid-way
- OSF auto-saves drafts. Reopen the draft from "My Registrations" and continue.
- If you accidentally submit before completing metadata, contact OSF support (support@osf.io) — they can move pending registrations back to draft state.

### If OSF rejects the registration
- Usually only happens if metadata violates OSF policies (PII, copyrighted material, etc.). Our bundle has none.
- If a specific file is flagged: redact the file, regenerate MANIFEST_SHA256, re-submit.

### If you discover a paper-grade bug AFTER OSF submission
- **CANNOT modify a published OSF registration** (this is the WHOLE POINT — immutable witness).
- Options:
  - (a) Document the bug in DOI 2 reproducibility bundle's `## known_issues_post_doi_1` section + paper §8 limitations.
  - (b) If the bug invalidates the witness itself (extremely rare): create a NEW OSF registration explicitly `supersedes` DOI 1 with corrected substrate. Both DOI strings remain visible on OSF.
- Standard scientific practice: small bugs go into DOI 2 errata; only fundamental issues warrant supersession.

### If Fire-3 fails / aborts after OSF submission
- DOI 1 is still valid — it claims "pre-data-creation witness", not "Pass-1 completed".
- Restart Fire-3 from clean state per [[phase1_plan]] §B; archive prior partial run_dir.
- DOI 2 (reproducibility bundle) reflects whatever data DOES land. If Fire-3 ultimately succeeds → DOI 2 normal mint. If Fire-3 abandoned → DOI 2 mints with `null_result` framing per [[negative_results_registry]].

## Cross-references

- Witness file: `docs/checkpoints/pre_run/artifact_existence_check_doi1_canonical_20260518T211628Z.txt`
- Deposit bundle: `docs/checkpoints/pre_run/osf_deposit_DOI1_20260518T211628Z/`
- Substance-lock anchor: tag `preregistration-locked-q3a` @ commit `72b93c9`
- Witness anchor: tag `preregistration-doi1-witnessed-20260518T211628Z` @ commit `5edac3b`
- OSF doctrine + workflow: `docs/checkpoints/pre_run/osf_lock_manifest.md` §3a
- Two-DOI split rationale: `docs/checkpoints/实验笔记.md` §230 + §233
- Fire-3 saga chronicle: `docs/checkpoints/实验笔记.md` §233

**Total user time**: ~15-30 min for steps 1-7 (manual). Then 0-48h passive wait for DOI assignment. Then ~10 min for step 9 backfill.

Good luck. The doctrine framework is solid; this is just clicking buttons in the right order.
