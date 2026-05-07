# Mirage Candidate Subset — classifieds (B1 Qwen3-VL-4B)

Extracted from `results/mechanistic/curate_mirage_b1_classifieds/candidates.jsonl` (笔记 §113, commit `cd50c34`). Used as paper-grade mirage dataset for Stage 2B curated scale-up + Stage 2C reverse-direction asymmetry confirm on A100.

## Filter criteria
- **Strong tier** (paper-grade mirage candidates): composite ≥ 1.0 AND token_overlap < 0.5
- **Reverse tier** (asymmetry robustness): composite ≤ -1.5

## Counts
- Strong: 24 tasks × 2 steps = up to 48 (task, step) artifacts
- Reverse: 15 tasks × 2 steps
- Skipped (no artifact): 0
- **Total disk**: 16.5 MB

## Strong candidates (top 24, sorted by composite desc)

| Rank | task_id | composite | overlap | intent (50 char) |
|---|---|---|---|---|
| 1 | 1 | +3.07 | 0.47 | Find me the cheapest red Toyota. It should be betw |
| 2 | 33 | +3.07 | 0.47 | Find the latest listing of a white fridge and post |
| 3 | 40 | +3.07 | 0.47 | Search for "dishwasher" and tell me the brand of t |
| 4 | 82 | +3.07 | 0.47 | Find the most expensive purple hard-case book, and |
| 5 | 116 | +3.07 | 0.47 | Search for "brace" and navigate to the cheapest it |
| 6 | 161 | +3.07 | 0.47 | Search for "mountain bike" and tell me the predomi |
| 7 | 214 | +3.07 | 0.47 | Show me the most expensive phone with a theme matc |
| 8 | 19 | +1.60 | 0.40 | Show me the most recently posted painting in the " |
| 9 | 20 | +1.60 | 0.40 | Show me the most recently posted white Xbox. |
| 10 | 122 | +1.60 | 0.40 | Find the oldest listed red book in the "Books" cat |
| 11 | 181 | +1.60 | 0.40 | Navigate to the most expensive item in the "Video  |
| 12 | 9 | +1.53 | 0.47 | Help me make a post selling this item and navigate |
| 13 | 32 | +1.53 | 0.47 | Find this exact bike that's listed for $300-500 an |
| 14 | 37 | +1.53 | 0.47 | Find all listings for this exact item on OSClass a |
| 15 | 60 | +1.53 | 0.47 | Find the most expensive video game item where I ca |
| 16 | 61 | +1.53 | 0.47 | Find the most expensive video game item where I ca |
| 17 | 69 | +1.53 | 0.47 | Find the most expensive item posted from Delaware  |
| 18 | 73 | +1.53 | 0.47 | Find me the cheapest posting from Delaware that ha |
| 19 | 99 | +1.53 | 0.47 | Find the most expensive wheelchair lift that has m |
| 20 | 108 | +1.53 | 0.47 | Tell me the name of the lister with the most expen |
| 21 | 109 | +1.53 | 0.47 | How many miles does the most expensive white sport |
| 22 | 215 | +1.53 | 0.47 | Show me the most expensive camera that is for sale |
| 23 | 227 | +1.53 | 0.47 | Show me the cheapest clock from the classifieds si |
| 24 | 228 | +1.53 | 0.47 | Find me the most recent live plant listing from th |

## Reverse candidates (sorted by composite asc)

| Rank | task_id | composite | overlap | intent (50 char) |
|---|---|---|---|---|
| 1 | 211 | -3.20 | 0.40 | Find me the cheapest farm pig in the "Farm + garde |
| 2 | 4 | -1.60 | 0.40 | Navigate to my listing of the white car and change |
| 3 | 152 | -1.60 | 0.40 | Navigate to the item on this page whose image incl |
| 4 | 159 | -1.60 | 0.40 | Navigate to my listing with a rug in the image and |
| 5 | 10 | -1.53 | 0.47 | What is the seat height in inches of the smaller p |
| 6 | 123 | -1.53 | 0.47 | Navigate to the item on this page which matches th |
| 7 | 130 | -1.53 | 0.47 | Navigate to the item on this page whose image is t |
| 8 | 151 | -1.53 | 0.47 | Navigate to the item on this page whose image has  |
| 9 | 155 | -1.53 | 0.47 | Explore the "Computers" category, and find the old |
| 10 | 156 | -1.53 | 0.47 | Explore the "Bikes" category, and find the oldest  |
| 11 | 157 | -1.53 | 0.47 | Explore the "Music instruments" category, and find |
| 12 | 160 | -1.53 | 0.47 | Add a comment on the item on this page whose image |
| 13 | 188 | -1.53 | 0.47 | Navigate to the book listing on the page where the |
| 14 | 191 | -1.53 | 0.47 | Navigate to the item on this page with a blonde ho |
| 15 | 200 | -1.53 | 0.47 | Show me the latest listing of a pillow from the cl |