# Mirage Candidate Subset — reddit (B1 Qwen3-VL-4B)

Extracted from `results/mechanistic/curate_mirage_b1_classifieds/candidates.jsonl` (笔记 §113, commit `cd50c34`). Used as paper-grade mirage dataset for Stage 2B curated scale-up + Stage 2C reverse-direction asymmetry confirm on A100.

## Filter criteria
- **Strong tier** (paper-grade mirage candidates): composite ≥ 1.0 AND token_overlap < 0.5
- **Reverse tier** (asymmetry robustness): composite ≤ -1.5

## Counts
- Strong: 47 tasks × 2 steps = up to 94 (task, step) artifacts
- Reverse: 48 tasks × 2 steps
- Skipped (no artifact): 0
- **Total disk**: 33.9 MB

## Strong candidates (top 24, sorted by composite desc)

| Rank | task_id | composite | overlap | intent (50 char) |
|---|---|---|---|---|
| 1 | 93 | +5.58 | 0.14 | What is the color of the plate used for the pictur |
| 2 | 116 | +3.76 | 0.12 | Navigate to the most commented post of all time (w |
| 3 | 79 | +3.68 | 0.16 | Navigate to the comments section of any post that  |
| 4 | 98 | +3.68 | 0.16 | Navigate to the comments section of a post that co |
| 5 | 99 | +3.68 | 0.16 | Navigate to the comments section of a post that co |
| 6 | 124 | +3.68 | 0.16 | Find me a post in /f/dataisbeautiful that illustra |
| 7 | 197 | +3.68 | 0.16 | Can you take me to the comments section of the mos |
| 8 | 78 | +3.60 | 0.20 | Navigate to the comments section of any post that  |
| 9 | 123 | +3.52 | 0.24 | Find me a post in /f/wallstreetbets that illustrat |
| 10 | 151 | +3.44 | 0.28 | Leave a comment in this post with the text as the  |
| 11 | 42 | +1.88 | 0.12 | Navigate to the comments section of the first imag |
| 12 | 65 | +1.88 | 0.12 | What is the total city population in 2021 of the t |
| 13 | 125 | +1.88 | 0.12 | Find me a post in /f/dataisbeautiful that illustra |
| 14 | 2 | +1.86 | 0.14 | I'm trying to find this post. Navigate to the comm |
| 15 | 4 | +1.86 | 0.14 | I'm trying to do some research about the post with |
| 16 | 18 | +1.86 | 0.14 | Navigate to a subreddit where I can find the most  |
| 17 | 32 | +1.86 | 0.14 | Find the user who posted this and navigate to the  |
| 18 | 115 | +1.86 | 0.14 | Navigate to the most controversial post of all tim |
| 19 | 147 | +1.86 | 0.14 | Leave a comment in this post with the text as the  |
| 20 | 156 | +1.86 | 0.14 | I saw a fun Mario and Princess Peach cosplay in f/ |
| 21 | 180 | +1.86 | 0.14 | Can you take me to the newest posts in the forum f |
| 22 | 194 | +1.86 | 0.14 | Can you comment 'This looks really fun to type on! |
| 23 | 207 | +1.86 | 0.14 | Can you first make a post in f/headphones (then af |
| 24 | 35 | +1.84 | 0.16 | Navigate to the comments section of the latest ima |

## Reverse candidates (sorted by composite asc)

| Rank | task_id | composite | overlap | intent (50 char) |
|---|---|---|---|---|
| 1 | 43 | -3.76 | 0.12 | Gather the title of the cheapest yellow video proj |
| 2 | 185 | -3.76 | 0.12 | Can you make a post in the /f/gaming forum titled  |
| 3 | 186 | -3.76 | 0.12 | Can you make a post in the /f/aww forum titled 'Pe |
| 4 | 106 | -3.68 | 0.16 | Subscribe to a forum that contains the following p |
| 5 | 172 | -3.68 | 0.16 | Can you post a comment on the post that has a cris |
| 6 | 103 | -3.64 | 0.18 | What is the color of the mattress? Leave a comment |
| 7 | 181 | -3.60 | 0.20 | I am a student at the school in the image. Can you |
| 8 | 199 | -3.52 | 0.24 | Can you comment 'I remember seeing him in person,  |
| 9 | 62 | -1.88 | 0.12 | Who potrayed the character in friends that has the |
| 10 | 158 | -1.88 | 0.12 | Can you subscribe to all subreddits that start wit |
| 11 | 165 | -1.88 | 0.12 | Can you find art of the building in the picture an |
| 12 | 208 | -1.88 | 0.12 | Can you first make a post in f/monitor (then after |
| 13 | 26 | -1.86 | 0.14 | Find this post and tell me what the top comment sa |
| 14 | 41 | -1.86 | 0.14 | Navigate to the comments section of the first imag |
| 15 | 47 | -1.86 | 0.14 | What is the area (in km^2) of the country that emi |
| 16 | 56 | -1.86 | 0.14 | What are the three major geographical divisions of |
| 17 | 63 | -1.86 | 0.14 | What is the highest elevation (in ft) of the home  |
| 18 | 67 | -1.86 | 0.14 | Who is the founder of the brand that has zero stor |
| 19 | 135 | -1.86 | 0.14 | Can you search for 'meme' and find the post with 2 |
| 20 | 142 | -1.86 | 0.14 | Leave a comment in this post with the release date |
| 21 | 164 | -1.86 | 0.14 | What is the top comment for the most popular post  |
| 22 | 195 | -1.86 | 0.14 | Can you comment 'Wish I had this for my setup' on  |
| 23 | 81 | -1.84 | 0.16 | Upvote all the posts that have a picture which con |
| 24 | 111 | -1.84 | 0.16 | Create a post for each of the following image in t |
| 25 | 112 | -1.84 | 0.16 | Create a post for each of the following image in t |
| 26 | 28 | -1.82 | 0.18 | Write a story about this picture and post it in th |
| 27 | 29 | -1.82 | 0.18 | Write a story about this picture and post it in th |
| 28 | 136 | -1.82 | 0.18 | Can you navigate to the post with 307 upvotes, whi |
| 29 | 49 | -1.80 | 0.20 | What is the first year the Summer Olympics include |
| 30 | 119 | -1.80 | 0.20 | What is the birth year of the guy holding the gobl |
| 31 | 122 | -1.80 | 0.20 | I'm doing some research about the post with this i |
| 32 | 148 | -1.80 | 0.20 | Leave a comment in this post with the text as the  |
| 33 | 6 | -1.78 | 0.22 | I'm trying to do some research about the post with |
| 34 | 80 | -1.78 | 0.22 | Upvote all the posts that have a picture which con |
| 35 | 83 | -1.78 | 0.22 | Upvote all the posts that have a picture which con |
| 36 | 177 | -1.78 | 0.22 | Can you comment on the food from the country in th |
| 37 | 183 | -1.78 | 0.22 | Can you make a post in f/food, title it '[I ate] t |
| 38 | 179 | -1.76 | 0.24 | Can you take me to the page that shows the most co |
| 39 | 182 | -1.76 | 0.24 | I am a student at the school in the image. Can you |
| 40 | 168 | -1.74 | 0.26 | Can you repost the image on this page that would f |
| 41 | 173 | -1.74 | 0.26 | Can you post a comment on the post with an 👽 in it |
| 42 | 74 | -1.72 | 0.28 | Find me a keyboard version of what the man is wear |
| 43 | 117 | -1.62 | 0.38 | What is the birth year of the dog with pink hair?  |
| 44 | 132 | -1.60 | 0.40 | How many comments did the post with this image rec |
| 45 | 145 | -1.60 | 0.40 | Leave a comment in this post with the text as the  |
| 46 | 59 | -1.56 | 0.44 | What is the total employees (in 2021) of the accou |
| 47 | 50 | -1.54 | 0.46 | What is the name of the major commercial airport i |
| 48 | 107 | -1.50 | 0.50 | Subscribe to a forum that contains the following p |