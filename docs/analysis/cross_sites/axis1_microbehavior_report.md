## Headline finding

Axis 1 decision-quality vs macro-frequency test (B0): B0 reddit=n/a cls=n/a; verdict: **not supported**.

## Per-(baseline, site) Axis 1 Table

| baseline | site | N | URL-path Jaccard | URL divergence | target-hit diff | target N | keyword repeat diff | distinct keyword diff | first-action divergence | macro mean | ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | reddit | 205 | 0.425 | 0.575 | 0.00 pp | 144 | 0.444 | -0.078 | 0.254 | n/a | n/a |
| B0 | classifieds | 224 | 0.849 | 0.151 | 0.00 pp | 172 | -0.397 | -0.174 | 0.308 | n/a | n/a |
| B1 | reddit | 205 | 0.509 | 0.491 | 0.00 pp | 144 | 0.717 | 0.088 | 0.244 | n/a | n/a |
| B1 | classifieds | 224 | 0.880 | 0.120 | 0.00 pp | 172 | 0.353 | 0.094 | 0.188 | n/a | n/a |
| B2 | reddit | — | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B2 | classifieds | 224 | 0.619 | 0.381 | 0.00 pp | 172 | 0.411 | 0.196 | 0.567 | n/a | n/a |

All signed differences are cascade-direction right-minus-left, so axis 1 is P-text minus DOM. Classifieds search-keyword levels should be read by axis differential, because OSClass tasks normally use search pages.

## Tier 1 Hook — Compound DOM ↔ P-SoM micro contrast

Direct test of the hook claim: even when aggregate macro action frequencies converge toward DOM (especially on classifieds, see Tier 1 macro), per-step decisions still diverge meaningfully. Lower URL-path Jaccard ⇒ more pages visited that DOM does not visit (or vice versa) ⇒ task-pool divergence at the decision-trace level.

| baseline | site | N | URL-path Jaccard (compound) | URL divergence | target-hit diff | first-action divergence |
|---|---|---:|---:|---:|---:|---:|
| B0 | reddit | 203 | 0.406 | 0.594 | 0.70 pp | 0.291 |
| B0 | classifieds | 224 | 0.863 | 0.137 | 0.00 pp | 0.281 |
| B1 | reddit | 205 | 0.434 | 0.566 | 0.00 pp | 0.371 |
| B1 | classifieds | 224 | 0.834 | 0.166 | 0.00 pp | 0.277 |
| B2 | reddit | — | n/a | n/a | n/a | n/a |
| B2 | classifieds | 224 | 0.611 | 0.389 | 0.00 pp | 0.621 |

## Cross-site Validity

Neither site clears the decision-over-macro ratio threshold (B0), so the paper should not claim that axis 1 primarily changes decision quality.

Validation checks: B0 axis-1 N is reddit 205/205 and classifieds 224/224. Target URLs were extracted for reddit 144/205 and classifieds 172/224 tasks. B1 axis-1 (P-text minus DOM) cannot be computed yet because B1 P-text data is pending; B1 compound (DOM ↔ P-SoM) is computed for cls only.

## Case Studies

### reddit task_23

Intent: Find this post and tell me what the top comment says.
Target: __REDDIT__/f/photoshopbattles/24816/-/comment/155650
URL-path Jaccard: 0.333
- DOM: steps=30, target_hit=False, reward=1.0, first=type -> /search, keywords=['pumpkin robot', 'pumpkin robot', 'pumpkin robot', 'pumpkin robot', 'pumpkin robot', 'pumpkin robot', 'pumpkin robot', 'pumpkin robot'], trajectory=/search?q=pumpkin+robot, /forums, /search?q=pumpkin+robot, /forums, /search?q=pumpkin+robot, /forums, /search?q=pumpkin+robot, /forums, /search?q=pumpkin+robot, /forums, /search?q=pumpkin+robot, /forums
- P-text: steps=10, target_hit=False, reward=0.0, first=type -> /search, keywords=['pumpkin robot', 'pumpkin'], trajectory=/search?q=pumpkin+robot, /search?q=pumpkin, /f/Art/10111/a-big-eater-me-pumpkin-carving-2022

### reddit task_30

Intent: Find the user who posted this and navigate to the comments section of one of their other posts.
Target: __REDDIT__/f/gaming/105429
URL-path Jaccard: 0.071
- DOM: steps=14, target_hit=False, reward=0.0, first=click -> /search, keywords=['colmscomics', 'colmscomics', 'colmscomics'], trajectory=/forums/all, /search?q=COLMSCOMICS, /forums/all, /search?q=COLMSCOMICS, /forums, /search?q=COLMSCOMICS, /user/MarvelsGrantMan136, /user/MarvelsGrantMan136/submissions, /user/MarvelsGrantMan136/comments, /f/television/134942
- P-text: steps=30, target_hit=False, reward=1.0, first=type -> /, keywords=['colmscomics', 'colmscomics', 'colmscomics', 'colmscomics', 'colmscomics', 'colmscomics'], trajectory=/search?q=COLMSCOMICS, /, /search?q=COLMSCOMICS, /, /search?q=COLMSCOMICS, /, /search?q=COLMSCOMICS, /, /search?q=COLMSCOMICS, /, /search?q=COLMSCOMICS, /user/colmscomics

### reddit task_4

Intent: I'm trying to do some research about the post with this image. Tell me how many comments mention the word "wheat".
Target: __REDDIT__/f/OldSchoolCool/121626/in-1982-agnes-denes-cultivated-grew-and-harvested-a-two-acre
URL-path Jaccard: 0.200
- DOM: steps=19, target_hit=False, reward=0.0, first=type -> /search, keywords=['wheat field in city', 'wheat field in city'], trajectory=/search?q=wheat+field+in+city, /f/nosleep/44227/i-accidentally-created-artificial-intelligence-in-my, /search?q=wheat+field+in+city
- P-text: steps=30, target_hit=False, reward=1.0, first=type -> /search, keywords=['wheat field cityscape', 'wheat field cityscape', 'wheat field cityscape', 'wheat field cityscape', 'wheat field cityscape', 'wheat field cityscape', 'wheat field cityscape', 'wheat field cityscape'], trajectory=/search?q=wheat+field+cityscape, /forums/all, /search?q=wheat+field+cityscape, /forums, /search?q=wheat+field+cityscape, /forums, /search?q=wheat+field+cityscape, /forums, /search?q=wheat+field+cityscape, /forums, /search?q=wheat+field+cityscape, /search?q=wheat

### classifieds task_226

Intent: Show me the cheapest figurine from the classifieds site that is of the third animal from the left shown in the product image on OneStopMarket.
Target: __CLASSIFIEDS__/index.php?page=item&id=8964
URL-path Jaccard: 0.000
- DOM: steps=27, target_hit=False, reward=0.0, first=type -> /index.php, keywords=['figurine'], trajectory=/index.php?page=search&sPattern=figurine, /index.php?page=search&sPattern=figurine&sOrder=i_price&iOrderType=asc, /index.php?page=item&id=83138, /index.php?page=search&sPattern=figurine&sOrder=i_price&iOrderType=asc, /5pcs-30ml-cute-mini-portable-gel-silicone-hand-bottle-cartoon-empty-spray-bottle-cosmetic-container-refillable-bottle-hand-soap-bottle.html, /index.php?page=search&sPattern=figurine&sOrder=i_price&iOrderType=asc, /index.php?page=item&id=83138, /index.php?page=search&sPattern=figurine&sOrder=i_price&iOrderType=asc, /index.php?page=item&id=83138, /index.php?page=search&sPattern=figurine&sOrder=i_price&iOrderType=asc, /index.php?page=item&id=83138, /index.php?page=search&sPattern=figurine&sOrder=i_price&iOrderType=asc
- P-text: steps=30, target_hit=False, reward=1.0, first=new_tab -> blank, keywords=['https://www.onestopmarket.com'], trajectory=blank

### classifieds task_199

Intent: Tell me the website mentioned in the image.
Target: None
URL-path Jaccard: 0.000
- DOM: steps=1, target_hit=None, reward=0.0, first=finish -> None, keywords=[], trajectory=/index.php?page=item&id=43887
- P-text: steps=2, target_hit=None, reward=0.0, first=goto -> /, keywords=[], trajectory=/

## Paper Section 5 Implication

The paper should rewrite the axis-1 claim: the micro-behavior evidence does not show a larger decision-quality effect than macro-frequency effect on either site.
