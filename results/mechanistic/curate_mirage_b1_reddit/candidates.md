# Mirage Task Candidates — reddit N=187

**Setup**: source `som` (with image) vs target `phantom_som` (no image), step 002, max_new_tokens=50.

**Composite score**: `(src_neg - tgt_neg) + (tgt_aff - src_aff)` × `(1 + divergence)`. Higher = stronger mirage axis (source 看到 ground-truth 无 → 否定 / target 幻觉 → 肯定).

**Token overlap baseline ~0.47-0.60** (from §111 envelope-only) — entries below 0.4 are real content divergence.

**Manual review**: pick top 10-20 entries with composite ≥ 1.0 AND overlap < 0.5 AND obvious mirage qualitative read.

## Top candidates (sorted by composite, descending)

| rank | task_id | composite | mirage_score | overlap | source | target | intent |
|---|---|---|---|---|---|---|---|
| 1 | 93 | +5.58 | +3 (sn3/sa0 / tn1/ta1) | 0.14 | `'{\n  "thought": "The search for \'f/food\' returned no results,'` | `'{\n  "thought": "The task requires finding the most controver'` | What is the color of the plate used for the picture from the |
| 2 | 116 | +3.76 | +2 (sn1/sa0 / tn1/ta2) | 0.12 | `'{\n  "thought": "The search for f/aww returned no results. Ac'` | `'{\n  "thought": "I am on the search page with a query for f/a'` | Navigate to the most commented post of all time (with a pict |
| 3 | 79 | +3.68 | +2 (sn1/sa1 / tn0/ta2) | 0.16 | `'{\n  "thought": "The current page shows recent comments, but '` | `'{\n  "thought": "The task is to navigate to the comments sect'` | Navigate to the comments section of any post that contains R |
| 4 | 98 | +3.68 | +2 (sn0/sa0 / tn0/ta2) | 0.16 | `'{\n  "thought": "The search bar is focused and ready for inpu'` | `'{\n  "thought": "The task is to navigate to the comments sect'` | Navigate to the comments section of a post that contains a p |
| 5 | 99 | +3.68 | +2 (sn1/sa1 / tn0/ta2) | 0.16 | `'{\n  "thought": "The current page shows recent comments, but '` | `'{\n  "thought": "The task is to navigate to the comments sect'` | Navigate to the comments section of a post that contains a p |
| 6 | 124 | +3.68 | +2 (sn2/sa0 / tn0/ta0) | 0.16 | `'{\n  "thought": "The search for \'/f/dataisbeautiful\' returned'` | `'{\n  "thought": "The search bar is focused and ready for inpu'` | Find me a post in /f/dataisbeautiful that illustrates the Mi |
| 7 | 197 | +3.68 | +2 (sn1/sa0 / tn0/ta1) | 0.16 | `'{\n  "thought": "The search for \'f/food\' returned no results,'` | `'{\n  "thought": "I am on the search page with a search box fo'` | Can you take me to the comments section of the most commente |
| 8 | 78 | +3.60 | +2 (sn1/sa1 / tn0/ta2) | 0.20 | `'{\n  "thought": "The task requires finding a post that contai'` | `'{\n  "thought": "The task is to navigate to the comments sect'` | Navigate to the comments section of any post that contains a |
| 9 | 123 | +3.52 | +2 (sn1/sa0 / tn0/ta1) | 0.24 | `'{\n  "thought": "The search query is currently set to \'/f/wal'` | `'{\n  "thought": "I am on the search page for /f/wallstreetbet'` | Find me a post in /f/wallstreetbets that illustrates the Ama |
| 10 | 151 | +3.44 | +2 (sn2/sa1 / tn0/ta1) | 0.28 | `'{\n  "thought": "The task requires leaving a comment with the'` | `'{\n  "thought": "The task requires leaving a comment with the'` | Leave a comment in this post with the text as the number of  |
| 11 | 42 | +1.88 | +1 (sn1/sa0 / tn1/ta1) | 0.12 | `'{\n  "thought": "The task requires navigating to the comments'` | `'{\n  "thought": "I am on the post page of the first image in '` | Navigate to the comments section of the first image post in  |
| 12 | 65 | +1.88 | +1 (sn0/sa0 / tn1/ta2) | 0.12 | `'{\n  "thought": "I need to access the Wikipedia site to find '` | `'{\n  "thought": "The current page is a \'Page not found\' error'` | What is the total city population in 2021 of the two cities  |
| 13 | 125 | +1.88 | +1 (sn1/sa0 / tn1/ta1) | 0.12 | `'{\n  "thought": "The search for \'/f/dataisbeautiful\' returned'` | `'{\n  "thought": "I am on the search results page for \'/f/data'` | Find me a post in /f/dataisbeautiful that illustrates the Nv |
| 14 | 2 | +1.86 | +1 (sn0/sa1 / tn0/ta2) | 0.14 | `'{\n  "thought": "The task is to navigate to the comment secti'` | `'{\n  "thought": "I am on the comments page for /f/movies, and'` | I'm trying to find this post. Navigate to the comment sectio |
| 15 | 4 | +1.86 | +1 (sn0/sa1 / tn0/ta2) | 0.14 | `'{\n  "thought": "The current screenshot shows a box of Wheat '` | `'{\n  "thought": "The image is displayed, but no comments sect'` | I'm trying to do some research about the post with this imag |
| 16 | 18 | +1.86 | +1 (sn0/sa1 / tn0/ta2) | 0.14 | `'{\n  "thought": "The task is to navigate to a subreddit with '` | `'{\n  "thought": "The current page is displaying comments in t'` | Navigate to a subreddit where I can find the most similar co |
| 17 | 32 | +1.86 | +1 (sn1/sa0 / tn1/ta1) | 0.14 | `'{\n  "thought": "The search returned no results, so I need to'` | `'{\n  "thought": "The task is to find the user who posted the '` | Find the user who posted this and navigate to the comments s |
| 18 | 115 | +1.86 | +1 (sn1/sa0 / tn1/ta1) | 0.14 | `'{\n  "thought": "The search for f/Paterson returned no result'` | `'{\n  "thought": "The current page shows a search for \'f/Pater'` | Navigate to the most controversial post of all time (with a  |
| 19 | 147 | +1.86 | +1 (sn0/sa0 / tn0/ta1) | 0.14 | `'{\n  "thought": "I need to click on the \'82 comments\' link to'` | `'{\n  "thought": "The task requires leaving a comment with the'` | Leave a comment in this post with the text as the number of  |
| 20 | 156 | +1.86 | +1 (sn1/sa0 / tn1/ta1) | 0.14 | `'{\n  "thought": "The search for \'f/gaming\' returned no result'` | `'{\n  "thought": "The user wants the top comment for a Mario a'` | I saw a fun Mario and Princess Peach cosplay in f/gaming. Ca |
| 21 | 180 | +1.86 | +1 (sn0/sa0 / tn0/ta1) | 0.14 | `'{\n  "thought": "The highlighted county in the image is Essex'` | `'{\n  "thought": "The task is to navigate to the newest posts '` | Can you take me to the newest posts in the forum for the hig |
| 22 | 194 | +1.86 | +1 (sn0/sa1 / tn0/ta2) | 0.14 | `'{\n  "thought": "The current screen shows a keyboard image bu'` | `'{\n  "thought": "The task requires commenting \'This looks rea'` | Can you comment 'This looks really fun to type on!' on the p |
| 23 | 207 | +1.86 | +1 (sn0/sa2 / tn0/ta3) | 0.14 | `'{\n  "thought": "The f/headphones forum is not visible in the'` | `'{\n  "thought": "I need to navigate to the \'headphones\' forum'` | Can you first make a post in f/headphones (then afterwards i |
| 24 | 35 | +1.84 | +1 (sn0/sa1 / tn0/ta2) | 0.16 | `'{\n  "thought": "The task requires navigating to the comments'` | `'{\n  "thought": "The current view shows the image itself, not'` | Navigate to the comments section of the latest image post in |
| 25 | 77 | +1.84 | +1 (sn0/sa1 / tn0/ta2) | 0.16 | `'{\n  "thought": "The task is to navigate to the comments sect'` | `'{\n  "thought": "The search results show a post titled \'MonaL'` | Navigate to the comments section of any post that contains M |
| 26 | 137 | +1.84 | +1 (sn1/sa0 / tn1/ta1) | 0.16 | `'{\n  "thought": "The search returned no results, so I need to'` | `'{\n  "thought": "The search query has been entered, but no re'` | Can you navigate to the post with 128 upvotes, which include |
| 27 | 162 | +1.84 | +1 (sn1/sa0 / tn0/ta0) | 0.16 | `'{\n  "thought": "The search for \'/f/wallstreetbets\' returned '` | `'{\n  "thought": "I need to search for a GIF related to the co'` | Can you give me the URL for a gif I could send to my friend  |
| 28 | 167 | +1.84 | +1 (sn0/sa0 / tn0/ta1) | 0.16 | `'{\n  "thought": "The current image is of a cat, not a dog and'` | `'{\n  "thought": "The image of the dog and cat is visible, but'` | I am feeling sad today. Can you navigate to the comments sec |
| 29 | 24 | +1.82 | +1 (sn0/sa0 / tn0/ta1) | 0.18 | `'{\n  "thought": "The target post is the first search result t'` | `'{\n  "thought": "The task is to find the post titled \'I paint'` | Find this post and tell me what the top comment says. |
| 30 | 154 | +1.82 | +1 (sn2/sa1 / tn0/ta0) | 0.18 | `'{\n  "thought": "The search for \'f/DataIsBeautiful\' returned '` | `'{\n  "thought": "I need to navigate to the f/DataIsBeautiful '` | I think I saw a post in f/DataIsBeautiful that talked about  |
| 31 | 96 | +1.80 | +1 (sn0/sa1 / tn0/ta2) | 0.20 | `'{\n  "thought": "I need to navigate to the f/Newark forum to '` | `'{\n  "thought": "I need to find the most commented post in f/'` | Who authored the post shown in the picture (not the reddit a |
| 32 | 100 | +1.78 | +1 (sn3/sa3 / tn0/ta1) | 0.22 | `'{\n  "thought": "The task is to find a post containing a pict'` | `'{\n  "thought": "The task is to navigate to the comments sect'` | Navigate to the comments section of a post that contains a p |
| 33 | 113 | +1.78 | +1 (sn0/sa0 / tn0/ta1) | 0.22 | `'{\n  "thought": "The task requires creating a post for each o'` | `'{\n  "thought": "The task is to create a post for each image '` | Create a post for each of the following image in the most re |
| 34 | 176 | +1.78 | +1 (sn0/sa1 / tn0/ta2) | 0.22 | `'{\n  "thought": "The task requires commenting on a post about'` | `'{\n  "thought": "The task requires commenting on Japanese foo'` | Can you comment on the food from the country in the image an |
| 35 | 72 | +1.76 | +1 (sn0/sa1 / tn0/ta2) | 0.24 | `'{\n  "thought": "The task is to make a comment on the post. T'` | `'{\n  "thought": "The task is to comment on the post, but no c'` | Make a comment in this post explaining what the picture is a |
| 36 | 166 | +1.74 | +1 (sn0/sa0 / tn0/ta1) | 0.26 | `'{\n  "thought": "The search result contains a post discussing'` | `'{\n  "thought": "The search result contains a post with the r'` | Can you find a post of the International Space Station's sol |
| 37 | 39 | +1.72 | +1 (sn0/sa0 / tn0/ta1) | 0.28 | `'{\n  "thought": "The user wants to find the subreddit for San'` | `'{\n  "thought": "The task is to find the subreddit for the ci'` | I'd like to find the subreddit for the city this photo was t |
| 38 | 189 | +1.72 | +1 (sn0/sa0 / tn0/ta1) | 0.28 | `'{\n  "thought": "The task is to subscribe to the most suitabl'` | `'{\n  "thought": "The user wants to subscribe to the most suit'` | Can you subscribe me to the most suitable forum where I can  |
| 39 | 188 | +1.70 | +1 (sn0/sa0 / tn0/ta1) | 0.30 | `'{\n  "thought": "The task is to subscribe to the most suitabl'` | `'{\n  "thought": "The user wants to subscribe to the most suit'` | Can you subscribe me to the most suitable forum where I can  |
| 40 | 190 | +1.70 | +1 (sn0/sa0 / tn0/ta1) | 0.30 | `'{\n  "thought": "The task is to subscribe to the most suitabl'` | `'{\n  "thought": "The user wants to subscribe to the most suit'` | Can you subscribe me to the most suitable forum where I can  |
| 41 | 73 | +1.68 | +1 (sn1/sa1 / tn0/ta1) | 0.32 | `'{\n  "thought": "The task requires finding a post in f/Mechan'` | `'{\n  "thought": "I need to find a post in f/MechanicalKeyboar'` | Find a post in f/MechanicalKeyboards that contains a special |
| 42 | 8 | +1.66 | +1 (sn0/sa0 / tn0/ta1) | 0.34 | `'{\n  "thought": "The task is to navigate to the comment URL o'` | `'{\n  "thought": "The task is to navigate to the comment URL o'` | I'm doing some research about the post with this image. Navi |
| 43 | 133 | +1.66 | +1 (sn0/sa0 / tn0/ta1) | 0.34 | `'{\n  "thought": "The task is to find the number of comments f'` | `'{\n  "thought": "The task is to find the number of comments o'` | How many comments did the post with this image receive? I re |
| 44 | 200 | +1.60 | +1 (sn0/sa1 / tn0/ta2) | 0.40 | `'{\n  "thought": "The task is to navigate to the post most rel'` | `'{\n  "thought": "The task is to navigate to the post most rel'` | Can you take me to the post on this page that is most relate |
| 45 | 37 | +1.56 | +1 (sn0/sa0 / tn0/ta1) | 0.44 | `'{\n  "thought": "The task is to navigate to the comments sect'` | `'{\n  "thought": "The task is to navigate to the comments sect'` | Navigate to the comments section of the latest image post in |
| 46 | 53 | +1.56 | +1 (sn1/sa1 / tn1/ta2) | 0.44 | `'{\n  "thought": "The current page is a \'Page not found\' error'` | `'{\n  "thought": "The current page is a \'Page not found\' error'` | What was the total asset value (in trillions of USD, to 3 de |
| 47 | 174 | +1.56 | +1 (sn1/sa1 / tn0/ta1) | 0.44 | `'{\n  "thought": "The task requires posting a comment with a 🛹'` | `'{\n  "thought": "The task requires posting a comment with a 🛹'` | Can you post a comment on the post with a 🛹 in it with the t |
| 48 | 193 | +1.48 | +1 (sn1/sa2 / tn0/ta2) | 0.52 | `'{\n  "thought": "The task requires commenting \'I love this!\' '` | `'{\n  "thought": "The task requires commenting \'I love this!\' '` | Can you comment 'I love this!' on the post with a blue/turqu |
| 49 | 20 | +1.42 | +1 (sn0/sa1 / tn0/ta2) | 0.58 | `'{\n  "thought": "The task is to find and upvote a post in /f/'` | `'{\n  "thought": "The task is to find and upvote a post in /f/'` | Find and upvote this post. I think it was posted in /f/space |
| 50 | 76 | +1.40 | +1 (sn0/sa0 / tn0/ta1) | 0.60 | `'{\n  "thought": "The task requires downvoting the post and co'` | `'{\n  "thought": "The task requires downvoting the post and co'` | How much money does the person have? Downvote the post and c |

## Bottom candidates (potential reverse-direction interest)

| rank | task_id | composite | mirage_score | overlap | source | target | intent |
|---|---|---|---|---|---|---|---|
| -1 | 142 | -1.86 | -1 | 0.14 | `'{\n  "thought": "The comment section is not directly accessib'` | `'{\n  "thought": "The task is to leave a comment with the rele'` | Leave a comment in this post with the release date of the mo |
| -2 | 164 | -1.86 | -1 | 0.14 | `'{\n  "thought": "I am on the \'All pages\' page of the Wiki, wh'` | `'{\n  "thought": "I need to find the most popular post of all '` | What is the top comment for the most popular post of all tim |
| -3 | 195 | -1.86 | -1 | 0.14 | `'{\n  "thought": "The target post with a pink escape key poste'` | `'{\n  "thought": "The task requires commenting \'Wish I had thi'` | Can you comment 'Wish I had this for my setup' on the post w |
| -4 | 62 | -1.88 | -1 | 0.12 | `'{\n  "thought": "I am on the Wiki page, but the task requires'` | `'{\n  "thought": "The current page is a \'Page not found\' error'` | Who potrayed the character in friends that has the most inti |
| -5 | 158 | -1.88 | -1 | 0.12 | `'{\n  "thought": "The task requires navigating to the \'Forums\''` | `'{\n  "thought": "I need to find all subreddits starting with '` | Can you subscribe to all subreddits that start with the lett |
| -6 | 165 | -1.88 | -1 | 0.12 | `'{\n  "thought": "I am on the notifications page, and I need t'` | `'{\n  "thought": "The task requires finding art of a building '` | Can you find art of the building in the picture and message  |
| -7 | 208 | -1.88 | -1 | 0.12 | `'{\n  "thought": "The f/monitor forum is not visible in the cu'` | `'{\n  "thought": "I need to navigate to the \'monitor\' section,'` | Can you first make a post in f/monitor (then afterwards in f |
| -8 | 199 | -3.52 | -2 | 0.24 | `'{\n  "thought": "The task requires navigating to the f/music '` | `'{\n  "thought": "The task requires commenting on the top post'` | Can you comment 'I remember seeing him in person, one of the |
| -9 | 181 | -3.60 | -2 | 0.20 | `'{\n  "thought": "The search results show a post from \'marche_'` | `'{\n  "thought": "The search query \'best pizza in [my school c'` | I am a student at the school in the image. Can you make a po |
| -10 | 103 | -3.64 | -2 | 0.18 | `'{\n  "thought": "The task requires identifying the color of t'` | `'{\n  "thought": "The task requires replying to the user \'BiAd'` | What is the color of the mattress? Leave a comment replying  |
| -11 | 106 | -3.68 | -2 | 0.16 | `'{\n  "thought": "The task requires subscribing to a forum tha'` | `'{\n  "thought": "The task is to subscribe to a forum that con'` | Subscribe to a forum that contains the following picture. |
| -12 | 172 | -3.68 | -2 | 0.16 | `'{\n  "thought": "The task is to post a comment on the post ab'` | `'{\n  "thought": "The task requires posting a comment on a pos'` | Can you post a comment on the post that has a crispy chicken |
| -13 | 43 | -3.76 | -2 | 0.12 | `'{\n  "thought": "The task requires finding the cheapest yello'` | `'{\n  "thought": "I am on the search results page for \'Video P'` | Gather the title of the cheapest yellow video projector in t |
| -14 | 185 | -3.76 | -2 | 0.12 | `'{\n  "thought": "The search results show no entries for \'/f/g'` | `'{\n  "thought": "I need to navigate to the /f/gaming forum. T'` | Can you make a post in the /f/gaming forum titled 'Should I  |
| -15 | 186 | -3.76 | -2 | 0.12 | `'{\n  "thought": "The search results show no entries for /f/aw'` | `'{\n  "thought": "I need to navigate to the /f/aww forum. The '` | Can you make a post in the /f/aww forum titled 'Perfect for  |

## Summary stats

- N candidates: 187
- skipped (missing artifacts): 23
- **Strong mirage candidates** (composite ≥ 1.0 AND overlap < 0.5): **47**
- composite distribution: min=-3.76, median=+0.00, max=+5.58