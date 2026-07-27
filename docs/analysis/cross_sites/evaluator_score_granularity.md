# Evaluator score granularity — landed Phase-1a conditions

Regenerate: `python3 scripts/analysis/aggregate_evaluator_score_granularity.py`

Universe: `grade: paper-grade` cells of `results/phantom_paper/run_manifest.yaml`.


**36 conditions · 7722 scored episodes.**


Distinct evaluator `score` values observed: **2** (0, 1).


| score | episodes | share |
|---|---|---|
| 0 | 7055 | 91.36% |
| 1 | 667 | 8.64% |

The evaluator is binary, so no graded quality target exists to regress on. This is a property of the benchmark's evaluation design, not of our pipeline.


Archived (pre-fix) conditions, reported separately and never pooled into the above: 3 conditions, 3 scored episodes, distinct values 0.


## Per condition

| condition | episodes | score 0 | score 1 |
|---|---|---|---|
| B0_DOM_classifieds | 224 | 185 | 39 |
| B0_DOM_reddit | 205 | 175 | 30 |
| B0_P-SoM_classifieds | 224 | 189 | 35 |
| B0_P-SoM_reddit | 205 | 182 | 23 |
| B0_P-prompt_classifieds | 224 | 180 | 44 |
| B0_P-prompt_reddit | 205 | 179 | 26 |
| B0_P-text_classifieds | 224 | 189 | 35 |
| B0_P-text_reddit | 205 | 177 | 28 |
| B0_SoM_classifieds | 224 | 163 | 61 |
| B0_SoM_reddit | 205 | 175 | 30 |
| B0_Vision_classifieds | 224 | 168 | 56 |
| B0_Vision_reddit | 205 | 189 | 16 |
| B1_DOM_classifieds | 224 | 210 | 14 |
| B1_DOM_reddit | 205 | 191 | 14 |
| B1_P-SoM_classifieds | 224 | 209 | 15 |
| B1_P-SoM_reddit | 205 | 191 | 14 |
| B1_P-prompt_classifieds | 224 | 209 | 15 |
| B1_P-prompt_reddit | 205 | 192 | 13 |
| B1_P-text_classifieds | 224 | 207 | 17 |
| B1_P-text_reddit | 205 | 191 | 14 |
| B1_SoM_classifieds | 224 | 192 | 32 |
| B1_SoM_reddit | 205 | 188 | 17 |
| B1_Vision_classifieds | 224 | 196 | 28 |
| B1_Vision_reddit | 205 | 199 | 6 |
| B2_DOM_classifieds | 224 | 221 | 3 |
| B2_DOM_reddit | 205 | 197 | 8 |
| B2_P-SoM_classifieds | 224 | 222 | 2 |
| B2_P-SoM_reddit | 205 | 202 | 3 |
| B2_P-prompt_classifieds | 224 | 220 | 4 |
| B2_P-prompt_reddit | 205 | 204 | 1 |
| B2_P-text_classifieds | 224 | 223 | 1 |
| B2_P-text_reddit | 205 | 200 | 5 |
| B2_SoM_classifieds | 224 | 219 | 5 |
| B2_SoM_reddit | 205 | 202 | 3 |
| B2_Vision_classifieds | 224 | 219 | 5 |
| B2_Vision_reddit | 205 | 200 | 5 |