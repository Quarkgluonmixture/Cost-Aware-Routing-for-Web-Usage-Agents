## Headline finding

Axis 1 decision-quality vs macro-frequency test: reddit ratio 2.42, classifieds ratio 1.46; verdict: **generalizes**.

## Per-site Axis 1 Table

| site | N | URL-path Jaccard | URL divergence | target-hit diff | target N | keyword repeat diff | distinct keyword diff | first-action divergence | macro mean | ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| reddit | 210 | 0.573 | 0.427 | 3.47 pp | 144 | -0.633 | 0.100 | 0.210 | 0.093 | 2.42 |
| classifieds | 234 | 0.904 | 0.096 | 2.33 pp | 172 | 0.077 | 0.004 | 0.137 | 0.059 | 1.46 |

All signed differences are cascade-direction right-minus-left, so axis 1 is P-DOM minus DOM. Classifieds search-keyword levels should be read by axis differential, because OSClass tasks normally use search pages.

## Tier 1 Hook — Compound DOM ↔ P-SoM micro contrast

Direct test of the hook claim: even when aggregate macro action frequencies converge toward DOM (especially on classifieds, see Tier 1 macro), per-step decisions still diverge meaningfully. Lower URL-path Jaccard ⇒ more pages visited that DOM does not visit (or vice versa) ⇒ task-pool divergence at the decision-trace level.

| site | N | URL-path Jaccard (compound) | URL divergence | target-hit diff | first-action divergence |
|---|---:|---:|---:|---:|---:|
| reddit | 210 | 0.481 | 0.519 | -0.69 pp | 0.248 |
| classifieds | 234 | 0.885 | 0.115 | 1.74 pp | 0.201 |

## Cross-site Validity

Both sites show a larger axis-1 shift in mode-invariant decision anchors than in macro action frequencies, so the claim generalizes beyond the reddit search-loop failure mode.

Validation checks: axis-1 N is reddit 210/210 and classifieds 234/234. Target URLs were extracted for reddit 144/210 and classifieds 172/234 tasks.

## Case Studies

### reddit task_23

Intent: Find this post and tell me what the top comment says.
Target: http://100.95.81.103:9999/f/photoshopbattles/24816/-/comment/155650
URL-path Jaccard: 0.500
- DOM: steps=30, target_hit=False, reward=1.0, first=type -> /search, keywords=['pumpkin robot', 'pumpkin', 'pumpkin robot', 'pumpkin', 'pumpkin robot', 'pumpkin', 'pumpkin robot', 'pumpkin'], trajectory=/search?q=pumpkin+robot+, /search?q=pumpkin+, /search?q=pumpkin+robot+, /search?q=pumpkin+, /search?q=pumpkin+robot+, /search?q=pumpkin+, /search?q=pumpkin+robot+, /search?q=pumpkin+, /search?q=pumpkin+robot+, /search?q=pumpkin+, /search?q=pumpkin+robot+, /search?q=pumpkin+
- Phantom-DOM: steps=10, target_hit=False, reward=1.0, first=type -> /search, keywords=['pumpkin robot', 'pumpkin', 'pumpkin robot', 'pumpkin'], trajectory=/search?q=pumpkin+robot+, /search?q=pumpkin+, /search?q=pumpkin+robot+, /search?q=pumpkin+, /submission_images/1b4655e4790380ba91d7eb446a736383e44dfcb3f41fdc7779c0f06610666ba4.jpg

### reddit task_30

Intent: Find the user who posted this and navigate to the comments section of one of their other posts.
Target: http://100.95.81.103:9999/f/gaming/105429
URL-path Jaccard: 0.500
- DOM: steps=30, target_hit=False, reward=1.0, first=type -> /, keywords=['colmscomics', 'colmscomics', 'good lord! this man has handgun ammo in his torso!', 'colmscomics', 'good lord! this man has handgun ammo in his torso!', 'colmscomics', 'good lord! this man has handgun ammo in his torso!', 'colmscomics'], trajectory=/search?q=COLMSCOMICS+, /, /search?q=COLMSCOMICS+, /search?q=Good+Lord%21+This+man+has+Handgun+ammo+in+his+torso%21+, /search?q=COLMSCOMICS+, /search?q=Good+Lord%21+This+man+has+Handgun+ammo+in+his+torso%21+, /search?q=COLMSCOMICS+, /search?q=Good+Lord%21+This+man+has+Handgun+ammo+in+his+torso%21+, /search?q=COLMSCOMICS+, /search?q=Good+Lord%21+This+man+has+Handgun+ammo+in+his+torso%21+, /search?q=COLMSCOMICS+, /search?q=Good+Lord%21+This+man+has+Handgun+ammo+in+his+torso%21+
- Phantom-DOM: steps=10, target_hit=False, reward=0.0, first=type -> /search, keywords=['colmscomics', 'colmscomics comic', 'colmscomics', 'colmscomics comic', 'colmscomics', 'colmscomics', 'user:colmscomics'], trajectory=/search?q=COLMSCOMICS+, /search?q=COLMSCOMICS+comic+, /search?q=COLMSCOMICS+, /search?q=COLMSCOMICS+comic+, /search?q=COLMSCOMICS+, /search?q=user%3ACOLMSCOMICS+

### reddit task_4

Intent: I'm trying to do some research about the post with this image. Tell me how many comments mention the word "wheat".
Target: http://100.95.81.103:9999/f/OldSchoolCool/121626/in-1982-agnes-denes-cultivated-grew-and-harvested-a-two-acre
URL-path Jaccard: 0.500
- DOM: steps=21, target_hit=True, reward=1.0, first=type -> /search, keywords=['wheat field city skyline', 'woman in wheat field city skyline', 'wheat field city skyline', 'woman in wheat field city skyline', 'wheat field city skyline', 'woman in wheat field city skyline', 'wheat field city skyline', 'woman in wheat field city skyline'], trajectory=/search?q=wheat+field+city+skyline+, /search?q=woman+in+wheat+field+city+skyline+, /search?q=wheat+field+city+skyline+, /search?q=woman+in+wheat+field+city+skyline+, /search?q=wheat+field+city+skyline+, /search?q=woman+in+wheat+field+city+skyline+, /search?q=wheat+field+city+skyline+, /search?q=woman+in+wheat+field+city+skyline+, /search?q=wheat+field+city+skyline+, /search?q=woman+wheat+field+city+skyline+, /search?q=wheat+field+city+skyline+, /search?q=woman+wheat+field+city+skyline+
- Phantom-DOM: steps=4, target_hit=False, reward=1.0, first=type -> /search, keywords=['wheat field woman'], trajectory=/search?q=wheat+field+woman+

### classifieds task_12

Intent: What is the color of most recently listed motorcycle?
Target: http://100.95.81.103:9980/index.php?page=item&id=9068
URL-path Jaccard: 0.500
- DOM: steps=9, target_hit=False, reward=0.0, first=type -> /index.php, keywords=['motorcycle'], trajectory=/index.php?page=search&sPattern=motorcycle+, /index.php?page=item&id=32241
- Phantom-DOM: steps=10, target_hit=True, reward=0.0, first=select_option -> /index.php, keywords=[], trajectory=/, /index.php?page=search&sCategory=20, /index.php?page=item&id=9068

### classifieds task_115

Intent: Search for "brace" and navigate to the most recently listed item that matches the body part in this image.
Target: http://100.95.81.103:9980/index.php?page=item&id=68133
URL-path Jaccard: 0.500
- DOM: steps=30, target_hit=False, reward=1.0, first=type -> /index.php, keywords=['foot brace', 'foot brace', 'foot brace', 'foot brace', 'foot brace', 'foot brace', 'foot brace'], trajectory=/index.php?page=search&sPattern=foot+brace+, /index.php?page=item&id=67190, /index.php?page=search&sPattern=foot+brace+, /index.php?page=item&id=67190, /index.php?page=search&sPattern=foot+brace+, /index.php?page=item&id=67190, /index.php?page=search&sPattern=foot+brace+, /index.php?page=item&id=67190, /index.php?page=search&sPattern=foot+brace+, /index.php?page=item&id=67190, /index.php?page=search&sPattern=foot+brace+, /index.php?page=item&id=67190
- Phantom-DOM: steps=5, target_hit=True, reward=0.0, first=type -> /index.php, keywords=['brace'], trajectory=/index.php?page=search&sPattern=brace+, /index.php?page=item&id=68133

## Paper Section 5 Implication

The paper can state that axis 1 is first-order at the task-success level because it changes where the agent goes and which target pages it reaches, even when macro action frequencies barely move. The classifieds result is important because it separates this from a reddit-only search-loop explanation; axis 3 can still be emphasized as stronger for image-heavy OSClass listing inspection.
