---
type: analysis
status: rolling
purpose: how much of each cell's environment time is spent on the public internet
producer: scripts/analysis/offsite_navigation_audit.py
---

# Off-site navigation audit

Regenerate: `.venv/bin/python3 scripts/analysis/offsite_navigation_audit.py`

Postmill is a **link aggregator**: a post's title is an external URL, so an agent opening a trending thread can walk onto the live public internet and keep acting there. The classifieds site is self-contained and offers almost no such exit.

A step is **off-site** when its `obs_url` host is not `localhost`. `env_step` is the environment-interaction slice of `latency_ms` — the part that contains page load.

## 1. Per cell

| cell | off-site steps | off-site episodes | median `env_step` on-site | off-site | ratio |
|---|---|---|---|---|---|
| `B0·VWA-cla` | 0/20646 (0.00%) | 0/1344 (0.0%) | 4,493 ms | — | — |
| `B0·VWA-red` | 478/26425 (1.81%) | 36/1230 (2.9%) | 11,349 ms | 10,007 ms | **0.88×** |
| `B1·VWA-cla` | 0/27927 (0.00%) | 0/1344 (0.0%) | 5,750 ms | — | — |
| `B1·VWA-red` | 501/29309 (1.71%) | 54/1230 (4.4%) | 7,848 ms | 6,349 ms | **0.81×** |
| `B2·VWA-cla` | 58/36529 (0.16%) | 4/1344 (0.3%) | 4,656 ms | 14,884 ms | **3.20×** |
| `B2·VWA-red` | 719/33749 (2.13%) | 49/1230 (4.0%) | 9,324 ms | 4,850 ms | **0.52×** |
| `B1·WA-red` | 278/14695 (1.89%) | 40/624 (6.4%) | 6,809 ms | 6,105 ms | **0.90×** |
| `B0·WA-red` | 123/11703 (1.05%) | 21/624 (3.4%) | 6,613 ms | 11,405 ms | **1.72×** |

## 2. The asymmetry

Classifieds: 0.00%, 0.00%, 0.16%. Reddit (VWA and WA): 1.81%, 1.71%, 2.13%, 1.89%, 1.05%.

**Off-site navigation is a reddit phenomenon.** That is a property of the application, not of any observation mode — but the paper's latency claim is stated at the *site* level (`multimetric_pareto`: the cheapest≠fastest split "follows the site, not the backbone"), and between-site latency comparisons therefore contain a component that is network geography rather than representation.

**The penalty runs the other way.** Off-site steps are *faster* than on-site ones in 4 of the 6 cells that have any (`B0·VWA-red` 0.88×, `B1·VWA-red` 0.81×, `B2·VWA-red` 0.52×, `B1·WA-red` 0.90×); slower in `B2·VWA-cla` 3.20×, `B0·WA-red` 1.72×. Commercial CDNs outrun a Postmill container sharing a host with the agent, so walking off-site buys time rather than costing it.

Largest distortion by exposure × magnitude: `B2·VWA-red`, 0.52× on 2.13% of steps — about **1.02%** of that cell's environment time, and in the direction that makes reddit look faster than it is.

Too small to overturn a latency ordering on its own. It is recorded because it is **undisclosed and one-sided** — it touches reddit and not classifieds, on the same axis as the site split the claim rests on.

## 2b. The larger asymmetry is the containers themselves

| site | median on-site `env_step` |
|---|---|
| `B0·VWA-cla` | 4,493 ms |
| `B0·VWA-red` | 11,349 ms |
| `B1·VWA-cla` | 5,750 ms |
| `B1·VWA-red` | 7,848 ms |
| `B2·VWA-cla` | 4,656 ms |
| `B2·VWA-red` | 9,324 ms |
| `B1·WA-red` | 6,809 ms |
| `B0·WA-red` | 6,613 ms |

Reddit's on-site page interaction costs 1.69× what classifieds' does, before any agent behaviour enters. This dwarfs the off-site effect above and is a property of Postmill versus Osclass on this host. It does **not** threaten the latency claim, which compares modes *within* a cell — but it does mean the phrase "follows the site" is carrying infrastructure as well as workload, and a between-site latency number should never be quoted bare.

## 3. Where they go

- `B0·VWA-red`: consent.yahoo.com (106), www.hollywoodreporter.com (82), www.theguardian.com (78), www.cnbc.com (52), www.yahoo.com (25)
- `B1·VWA-red`: postmill.xyz (147), www.bobvila.com (58), www.theguardian.com (58), consent.yahoo.com (41), www.archdaily.com (29)
- `B2·VWA-cla`: osclass-classifieds.com (47), www.cloudflare.com (8), osclasspoint.com (2), www.onetrust.com (1)
- `B2·VWA-red`: postmill.xyz (261), www.wfsb.com (96), newsroom.kelloggcompany.com (41), www.cloudflare.com (40), omnitype.com (32)
- `B1·WA-red`: imgur.com (46), www.supercluster.com (29), eu.detroitnews.com (27), help.imgur.com (25), deadline.com (23)
- `B0·WA-red`: www.supercluster.com (68), www.publishersweekly.com (15), spaceagency.supercluster.com (12), imgur.com (6), www.publichealth.columbia.edu (6)

⚠️ These are **live public sites reached from an experiment host**. Nothing was submitted to them — the actions are clicks and scrolls on pages the agent loaded — but the runs are not hermetic, and a replication on a network-isolated host would not reproduce these steps at all.

