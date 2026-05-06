# Mirage Task Candidates — classifieds N=209

**Setup**: source `som` (with image) vs target `phantom_som` (no image), step 002, max_new_tokens=15.

**Composite score**: `(src_neg - tgt_neg) + (tgt_aff - src_aff)` × `(1 + divergence)`. Higher = stronger mirage axis (source 看到 ground-truth 无 → 否定 / target 幻觉 → 肯定).

**Token overlap baseline ~0.47-0.60** (from §111 envelope-only) — entries below 0.4 are real content divergence.

**Manual review**: pick top 10-20 entries with composite ≥ 1.0 AND overlap < 0.5 AND obvious mirage qualitative read.

## Top candidates (sorted by composite, descending)

| rank | task_id | composite | mirage_score | overlap | source | target | intent |
|---|---|---|---|---|---|---|---|
| 1 | 0 | +4.20 | +3 (sn2/sa1 / tn0/ta2) | 0.60 | `'{\n  "thought": "The search results do not show any blue kay'` | `'{\n  "thought": "The search results show items related to \'bl'` | Find me the cheapest blue kayak on this site. |
| 2 | 81 | +4.20 | +3 (sn2/sa1 / tn0/ta2) | 0.60 | `'{\n  "thought": "The search results do not show any listings '` | `'{\n  "thought": "The search results show listings for \'hur'` | Find the cheapest book with a hurricane on the cover. |
| 3 | 112 | +4.20 | +3 (sn2/sa1 / tn0/ta2) | 0.60 | `'{\n  "thought": "The search results do not show any item with'` | `'{\n  "thought": "The search results show items related to bas'` | Search for "basketball" and navigate to the cheapest item th |
| 4 | 113 | +4.20 | +3 (sn2/sa1 / tn0/ta2) | 0.60 | `'{\n  "thought": "The search results do not show any item with'` | `'{\n  "thought": "The search results show items related to foo'` | Search for "football" and navigate to the most expensive ite |
| 5 | 127 | +4.20 | +3 (sn2/sa1 / tn0/ta2) | 0.60 | `'{\n  "thought": "The search results do not show any item with'` | `'{\n  "thought": "The search results show listings for \'MCAT'` | Search for "MCAT" and navigate to the prep book that has 202 |
| 6 | 201 | +4.20 | +3 (sn2/sa1 / tn0/ta2) | 0.60 | `'{\n  "thought": "The search results do not show any snare'` | `'{\n  "thought": "The search results show listings for various'` | Show me the latest listing of a snare drum with a black and  |
| 7 | 224 | +4.00 | +3 (sn2/sa1 / tn0/ta2) | 0.67 | `'{\n  "thought": "The search results do not show any wall rack'` | `'{\n  "thought": "The search results show listings for \'wall r'` | Show me the cheapest wall rack between $30-40 from the class |
| 8 | 1 | +3.07 | +2 (sn0/sa0 / tn0/ta2) | 0.47 | `'{\n  "thought": "The task is to find the cheapest red Toyota'` | `'{\n  "thought": "The search results show a 200'` | Find me the cheapest red Toyota. It should be between $3000  |
| 9 | 33 | +3.07 | +2 (sn2/sa1 / tn0/ta1) | 0.47 | `'{\n  "thought": "The search results do not show any white fri'` | `'{\n  "thought": "The current page shows search results for \'w'` | Find the latest listing of a white fridge and post a comment |
| 10 | 40 | +3.07 | +2 (sn0/sa0 / tn0/ta2) | 0.47 | `'{\n  "thought": "The task is to find the most recent listing'` | `'{\n  "thought": "The search results show multiple dishwasher '` | Search for "dishwasher" and tell me the brand of the most re |
| 11 | 82 | +3.07 | +2 (sn0/sa0 / tn0/ta2) | 0.47 | `'{\n  "thought": "The task is to find the most expensive purpl'` | `'{\n  "thought": "The search results show listings for guitars'` | Find the most expensive purple hard-case book, and tell me h |
| 12 | 116 | +3.07 | +2 (sn0/sa0 / tn0/ta2) | 0.47 | `'{\n  "thought": "The task is to find the cheapest item that'` | `'{\n  "thought": "The search results show listings for \'should'` | Search for "brace" and navigate to the cheapest item that ma |
| 13 | 161 | +3.07 | +2 (sn0/sa0 / tn0/ta2) | 0.47 | `'{\n  "thought": "The task is to find the cheapest mountain bi'` | `'{\n  "thought": "The search results show multiple mountain bi'` | Search for "mountain bike" and tell me the predominant color |
| 14 | 214 | +3.07 | +2 (sn2/sa1 / tn0/ta1) | 0.47 | `'{\n  "thought": "The search results do not show any Star Wars'` | `'{\n  "thought": "The current page shows search results for \'S'` | Show me the most expensive phone with a theme matching that  |
| 15 | 17 | +2.93 | +2 (sn2/sa1 / tn0/ta1) | 0.53 | `'{\n  "thought": "The current listings do not show any bike wi'` | `'{\n  "thought": "The current page shows search results for bi'` | Show me the cheapest bike with red handlebars between $900-9 |
| 16 | 202 | +2.93 | +2 (sn0/sa0 / tn0/ta2) | 0.53 | `'{\n  "thought": "The task is to find the cheapest penguin'` | `'{\n  "thought": "The search results show listings for \'pengui'` | Show me the cheapest penguin-themed item in West Virginia. |
| 17 | 204 | +2.93 | +2 (sn2/sa1 / tn0/ta1) | 0.53 | `'{\n  "thought": "The current search results do not show any g'` | `'{\n  "thought": "The current page shows search results for gu'` | Find the cheapest guitar between $1200 and $1500 that is mad |
| 18 | 206 | +2.93 | +2 (sn2/sa1 / tn0/ta1) | 0.53 | `'{\n  "thought": "The current search results do not show any f'` | `'{\n  "thought": "The current page shows search results for \'f'` | Find the cheapest freezer with a warranty between $250 and $ |
| 19 | 89 | +2.80 | +2 (sn2/sa1 / tn1/ta2) | 0.60 | `'{\n  "thought": "The search results do not show any cars,'` | `'{\n  "thought": "The search results show listings, but none a'` | Find the most expensive car from Virginia that is neon green |
| 20 | 180 | +2.80 | +2 (sn2/sa1 / tn0/ta1) | 0.60 | `'{\n  "thought": "The search results do not show any coffee ma'` | `'{\n  "thought": "The search results are showing furniture and'` | Find the most recently listed coffee maker with a touch scre |
| 21 | 19 | +1.60 | +1 (sn0/sa0 / tn0/ta1) | 0.40 | `'{\n  "thought": "I need to refine the search to include \''` | `'{\n  "thought": "The current page shows search results for \'s'` | Show me the most recently posted painting in the "Arts + cra |
| 22 | 20 | +1.60 | +1 (sn0/sa0 / tn0/ta1) | 0.40 | `'{\n  "thought": "I am on a single item page for a'` | `'{\n  "thought": "The current page shows a listing for an Xbox'` | Show me the most recently posted white Xbox. |
| 23 | 122 | +1.60 | +1 (sn0/sa0 / tn0/ta1) | 0.40 | `'{\n  "thought": "The \'Books\' category is already selected,'` | `'{\n  "thought": "I need to find the oldest listed red book'` | Find the oldest listed red book in the "Books" category, and |
| 24 | 181 | +1.60 | +1 (sn0/sa0 / tn0/ta1) | 0.40 | `'{\n  "thought": "The task requires navigating to the most exp'` | `'{\n  "thought": "I am currently on the item page for \''` | Navigate to the most expensive item in the "Video gaming" ca |
| 25 | 9 | +1.53 | +1 (sn0/sa0 / tn0/ta1) | 0.47 | `'{\n  "thought": "The task requires finding the most similar i'` | `'{\n  "thought": "The current page is showing search results f'` | Help me make a post selling this item and navigate to it. Pr |
| 26 | 32 | +1.53 | +1 (sn0/sa0 / tn0/ta1) | 0.47 | `'{\n  "thought": "The target bike is a GT Stalker '` | `'{\n  "thought": "I need to find a bike listed for $'` | Find this exact bike that's listed for $300-500 and post a c |
| 27 | 37 | +1.53 | +1 (sn0/sa0 / tn0/ta1) | 0.47 | `'{\n  "thought": "The task is to find listings for a specific'` | `'{\n  "thought": "The current page is showing search results f'` | Find all listings for this exact item on OSClass and compare |
| 28 | 60 | +1.53 | +1 (sn0/sa0 / tn0/ta1) | 0.47 | `'{\n  "thought": "The task is to find the most expensive video'` | `'{\n  "thought": "The most expensive item visible is the \'g'` | Find the most expensive video game item where I can roleplay |
| 29 | 61 | +1.53 | +1 (sn0/sa0 / tn0/ta1) | 0.47 | `'{\n  "thought": "The task is to find the most expensive video'` | `'{\n  "thought": "The most expensive item visible is the \'g'` | Find the most expensive video game item where I can roleplay |
| 30 | 69 | +1.53 | +1 (sn0/sa0 / tn0/ta1) | 0.47 | `'{\n  "thought": "The task requires finding the most expensive'` | `'{\n  "thought": "The current page shows search results for \'N'` | Find the most expensive item posted from Delaware which is r |
| 31 | 73 | +1.53 | +1 (sn0/sa0 / tn0/ta1) | 0.47 | `'{\n  "thought": "The task requires finding the cheapest posti'` | `'{\n  "thought": "The current page shows search results for \'M'` | Find me the cheapest posting from Delaware that has the same |
| 32 | 99 | +1.53 | +1 (sn0/sa0 / tn0/ta1) | 0.47 | `'{\n  "thought": "The task requires finding the most expensive'` | `'{\n  "thought": "The current page shows listings for wheelcha'` | Find the most expensive wheelchair lift that has multiple pe |
| 33 | 108 | +1.53 | +1 (sn0/sa0 / tn0/ta1) | 0.47 | `'{\n  "thought": "The task is to find the lister with'` | `'{\n  "thought": "The current page shows search results for \'g'` | Tell me the name of the lister with the most expensive green |
| 34 | 109 | +1.53 | +1 (sn0/sa0 / tn0/ta1) | 0.47 | `'{\n  "thought": "The task is to find the most expensive white'` | `'{\n  "thought": "The current page shows search results for \'w'` | How many miles does the most expensive white sports car in t |
| 35 | 215 | +1.53 | +1 (sn0/sa0 / tn0/ta1) | 0.47 | `'{\n  "thought": "The task is to confirm the most expensive ca'` | `'{\n  "thought": "The most expensive camera listed is the Cano'` | Show me the most expensive camera that is for sale (as confi |
| 36 | 227 | +1.53 | +1 (sn0/sa0 / tn0/ta1) | 0.47 | `'{\n  "thought": "The task requires finding the cheapest clock'` | `'{\n  "thought": "The current page shows search results for \'c'` | Show me the cheapest clock from the classifieds site with th |
| 37 | 228 | +1.53 | +1 (sn0/sa0 / tn0/ta1) | 0.47 | `'{\n  "thought": "The task requires finding the most recent li'` | `'{\n  "thought": "The image shows a spider plant, and the'` | Find me the most recent live plant listing from the classifi |
| 38 | 64 | +1.47 | +1 (sn0/sa0 / tn0/ta1) | 0.53 | `'{\n  "thought": "The current search is filtered to \'Cars +'` | `'{\n  "thought": "The current page shows search results for \'v'` | Find the video game item that costs exactly $500 where I can |
| 39 | 68 | +1.47 | +1 (sn2/sa1 / tn0/ta0) | 0.53 | `'{\n  "thought": "The current search results do not show any T'` | `'{\n  "thought": "The current page is a search results page fo'` | Find the cheapest Tiger Woods video game item where the pers |
| 40 | 142 | +1.47 | +1 (sn1/sa0 / tn0/ta0) | 0.53 | `'{\n  "thought": "The search returned no results, but the task'` | `'{\n  "thought": "The search query \'8\\" abstract ocean paintin'` | I recall seeing this exact item for under $20 on the site, t |
| 41 | 146 | +1.47 | +1 (sn0/sa0 / tn0/ta1) | 0.53 | `'{\n  "thought": "The task requires navigating to the item who'` | `'{\n  "thought": "The current page shows an item with an image'` | Navigate to the item on this page whose image has exactly 4  |
| 42 | 198 | +1.47 | +1 (sn0/sa0 / tn0/ta1) | 0.53 | `'{\n  "thought": "The current search results are for cars, not'` | `'{\n  "thought": "The current page shows car listings, but the'` | Navigate to the cheapest watch with hints of faded red and g |
| 43 | 190 | +1.40 | +1 (sn0/sa0 / tn0/ta1) | 0.60 | `'{\n  "thought": "The current page is for a PlayStation PS1'` | `'{\n  "thought": "The current page shows a single video game i'` | Navigate to the cheapest video game item on this page with t |
| 44 | 217 | +1.40 | +1 (sn2/sa1 / tn0/ta0) | 0.60 | `'{\n  "thought": "The search results do not show any listing f'` | `'{\n  "thought": "The search results page is already filtered '` | Find me the most recent listing offering a book titled "Capt |
| 45 | 203 | +1.27 | +1 (sn0/sa0 / tn0/ta1) | 0.73 | `'{\n  "thought": "I am on the item detail page for the'` | `'{\n  "thought": "I am on the item page for the Se'` | Help me leave a comment with the title "Interested" with the |
| 46 | 219 | +1.13 | +1 (sn0/sa0 / tn0/ta1) | 0.87 | `'{\n  "thought": "The current item is a wall mirror coat h'` | `'{\n  "thought": "The current page shows a wall mirror coat h'` | Show me the cheapest coat hanger between $15 and $20 that ca |
| 47 | 2 | +0.00 | +0 (sn0/sa0 / tn0/ta0) | 0.47 | `'{\n  "thought": "I need to filter listings for red gem jewelr'` | `'{\n  "thought": "I am on the Jewelry category page. I'` | Find me the most recently posted item in the "Jewelry" categ |
| 48 | 3 | +0.00 | +0 (sn0/sa0 / tn0/ta0) | 0.47 | `'{\n  "thought": "I am on the \'Photo + video\''` | `'{\n  "thought": "I need to find the most recently posted item'` | Find me the most recently posted item in the "Photo + video" |
| 49 | 5 | +0.00 | +0 (sn0/sa0 / tn0/ta0) | 0.40 | `'{\n  "thought": "The task is to find and delete the white'` | `'{\n  "thought": "I am on the \'Manage my listings\''` | Navigate to my listing of the white car and delete it. |
| 50 | 6 | +0.00 | +0 (sn0/sa0 / tn0/ta0) | 0.60 | `'{\n  "thought": "I need to search for motorcycles within $1'` | `'{\n  "thought": "I need to find motorcycles within $10'` | Return the links of the 3 most recent motorcycles within $10 |

## Bottom candidates (potential reverse-direction interest)

| rank | task_id | composite | mirage_score | overlap | source | target | intent |
|---|---|---|---|---|---|---|---|
| -1 | 10 | -1.53 | -1 | 0.47 | `'{\n  "thought": "The current page is displaying a JBL speaker'` | `'{\n  "thought": "The task is to find the seat height in'` | What is the seat height in inches of the smaller piece of fu |
| -2 | 123 | -1.53 | -1 | 0.47 | `'{\n  "thought": "The search results are showing cars and truc'` | `'{\n  "thought": "The task is to navigate to the item that'` | Navigate to the item on this page which matches the color of |
| -3 | 130 | -1.53 | -1 | 0.47 | `'{\n  "thought": "The current page shows an item titled \'Wood'` | `'{\n  "thought": "The task is to navigate to the item whose'` | Navigate to the item on this page whose image is taken durin |
| -4 | 151 | -1.53 | -1 | 0.47 | `'{\n  "thought": "The item with pink sticky notes is visible i'` | `'{\n  "thought": "The task is to navigate to the item whose'` | Navigate to the item on this page whose image has pink stick |
| -5 | 155 | -1.53 | -1 | 0.47 | `'{\n  "thought": "The task requires finding the oldest listed '` | `'{\n  "thought": "I need to explore the \'Computers\''` | Explore the "Computers" category, and find the oldest listed |
| -6 | 156 | -1.53 | -1 | 0.47 | `'{\n  "thought": "The task requires finding the oldest listed '` | `'{\n  "thought": "I need to explore the \'Bikes\''` | Explore the "Bikes" category, and find the oldest listed ite |
| -7 | 157 | -1.53 | -1 | 0.47 | `'{\n  "thought": "The task requires finding the oldest listed '` | `'{\n  "thought": "I need to explore the \'Music instruments\''` | Explore the "Music instruments" category, and find the oldes |
| -8 | 160 | -1.53 | -1 | 0.47 | `'{\n  "thought": "The target image is not visible on the curre'` | `'{\n  "thought": "The task requires adding a comment to an ite'` | Add a comment on the item on this page whose image is taken  |
| -9 | 188 | -1.53 | -1 | 0.47 | `'{\n  "thought": "The current page displays books without a ba'` | `'{\n  "thought": "The task is to navigate to the book listing'` | Navigate to the book listing on the page where the cover inc |
| -10 | 191 | -1.53 | -1 | 0.47 | `'{\n  "thought": "The current page shows a listing for a \''` | `'{\n  "thought": "The task is to navigate to the item with'` | Navigate to the item on this page with a blonde horse. |
| -11 | 200 | -1.53 | -1 | 0.47 | `'{\n  "thought": "The page shows a 404 error'` | `'{\n  "thought": "The current page is a 404'` | Show me the latest listing of a pillow from the classifieds  |
| -12 | 4 | -1.60 | -1 | 0.40 | `'{\n  "thought": "The white car listing is not visible in the'` | `'{\n  "thought": "I am on the \'Manage my listings\''` | Navigate to my listing of the white car and change the price |
| -13 | 152 | -1.60 | -1 | 0.40 | `'{\n  "thought": "The item with a human hand is visible in'` | `'{\n  "thought": "I need to find an item whose image includes'` | Navigate to the item on this page whose image includes a hum |
| -14 | 159 | -1.60 | -1 | 0.40 | `'{\n  "thought": "The current page shows listings for cars, bi'` | `'{\n  "thought": "I am on the \'Manage my listings\''` | Navigate to my listing with a rug in the image and add to th |
| -15 | 211 | -3.20 | -2 | 0.40 | `'{\n  "thought": "The search results show listings for \'farm p'` | `'{\n  "thought": "I am on the search results page for \''` | Find me the cheapest farm pig in the "Farm + garden" categor |

## Summary stats

- N candidates: 209
- skipped (missing artifacts): 25
- **Strong mirage candidates** (composite ≥ 1.0 AND overlap < 0.5): **24**
- composite distribution: min=-3.20, median=+0.00, max=+4.20