## Headline finding

Axis 1 decision-quality vs macro-frequency test (B0): B0 reddit=n/a cls=n/a; verdict: **not supported**.

## Per-(baseline, site) Axis 1 Table

| baseline | site | N | URL-path Jaccard | URL divergence | target-hit diff | target N | keyword repeat diff | distinct keyword diff | first-action divergence | macro mean | ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | reddit | — | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B0 | classifieds | — | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B1 | reddit | — | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B1 | classifieds | — | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B2 | reddit | — | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B2 | classifieds | — | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

All signed differences are cascade-direction right-minus-left, so axis 1 is P-text minus DOM. Classifieds search-keyword levels should be read by axis differential, because OSClass tasks normally use search pages.

## Tier 1 Hook — Compound DOM ↔ P-SoM micro contrast

Direct test of the hook claim: even when aggregate macro action frequencies converge toward DOM (especially on classifieds, see Tier 1 macro), per-step decisions still diverge meaningfully. Lower URL-path Jaccard ⇒ more pages visited that DOM does not visit (or vice versa) ⇒ task-pool divergence at the decision-trace level.

| baseline | site | N | URL-path Jaccard (compound) | URL divergence | target-hit diff | first-action divergence |
|---|---|---:|---:|---:|---:|---:|
| B0 | reddit | — | n/a | n/a | n/a | n/a |
| B0 | classifieds | — | n/a | n/a | n/a | n/a |
| B1 | reddit | — | n/a | n/a | n/a | n/a |
| B1 | classifieds | — | n/a | n/a | n/a | n/a |
| B2 | reddit | — | n/a | n/a | n/a | n/a |
| B2 | classifieds | — | n/a | n/a | n/a | n/a |

## Cross-site Validity

Neither site clears the decision-over-macro ratio threshold (B0), so the paper should not claim that axis 1 primarily changes decision quality.

Validation checks: B0 axis-1 N is reddit 0/205 and classifieds 0/224. Target URLs were extracted for reddit 144/205 and classifieds 172/224 tasks. B1 axis-1 (P-text minus DOM) cannot be computed yet because B1 P-text data is pending; B1 compound (DOM ↔ P-SoM) is computed for cls only.

## Case Studies

## Paper Section 5 Implication

The paper should rewrite the axis-1 claim: the micro-behavior evidence does not show a larger decision-quality effect than macro-frequency effect on either site.
