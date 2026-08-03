# Failure modes per cell (paper §5 — 5-bucket taxonomy)

5-bucket paper taxonomy mapped from fine-grained reason_bucket (see `aggregate_failure_modes.py` PAPER_TAXONOMY).

## Per-cell breakdown

### B0/classifieds/DOM (N=224, failed=185)

| Paper bucket | Count | % of failed | % of total |
|---|---:|---:|---:|
| early-finish/wrong-commit | 129 | 69.7% | 57.6% |
| max-steps-other | 14 | 7.6% | 6.2% |
| search-loop | 27 | 14.6% | 12.1% |
| visual-hijack/click-loop | 15 | 8.1% | 6.7% |

### B0/classifieds/P-SoM (N=224, failed=189)

| Paper bucket | Count | % of failed | % of total |
|---|---:|---:|---:|
| early-finish/wrong-commit | 120 | 63.5% | 53.6% |
| max-steps-other | 24 | 12.7% | 10.7% |
| search-loop | 23 | 12.2% | 10.3% |
| visual-hijack/click-loop | 22 | 11.6% | 9.8% |

### B0/classifieds/P-prompt (N=224, failed=180)

| Paper bucket | Count | % of failed | % of total |
|---|---:|---:|---:|
| early-finish/wrong-commit | 118 | 65.6% | 52.7% |
| max-steps-other | 13 | 7.2% | 5.8% |
| search-loop | 24 | 13.3% | 10.7% |
| visual-hijack/click-loop | 25 | 13.9% | 11.2% |

### B0/classifieds/P-text (N=224, failed=189)

| Paper bucket | Count | % of failed | % of total |
|---|---:|---:|---:|
| early-finish/wrong-commit | 125 | 66.1% | 55.8% |
| max-steps-other | 14 | 7.4% | 6.2% |
| search-loop | 22 | 11.6% | 9.8% |
| visual-hijack/click-loop | 28 | 14.8% | 12.5% |

### B0/classifieds/SoM (N=224, failed=163)

| Paper bucket | Count | % of failed | % of total |
|---|---:|---:|---:|
| early-finish/wrong-commit | 105 | 64.4% | 46.9% |
| max-steps-other | 18 | 11.0% | 8.0% |
| search-loop | 22 | 13.5% | 9.8% |
| visual-hijack/click-loop | 18 | 11.0% | 8.0% |

### B0/classifieds/Vision (N=224, failed=168)

| Paper bucket | Count | % of failed | % of total |
|---|---:|---:|---:|
| early-finish/wrong-commit | 104 | 61.9% | 46.4% |
| max-steps-other | 39 | 23.2% | 17.4% |
| search-loop | 17 | 10.1% | 7.6% |
| visual-hijack/click-loop | 8 | 4.8% | 3.6% |

### B0/reddit/DOM (N=205, failed=175)

| Paper bucket | Count | % of failed | % of total |
|---|---:|---:|---:|
| early-finish/wrong-commit | 81 | 46.3% | 39.5% |
| max-steps-other | 42 | 24.0% | 20.5% |
| search-loop | 31 | 17.7% | 15.1% |
| visual-hijack/click-loop | 21 | 12.0% | 10.2% |

### B0/reddit/P-prompt (N=205, failed=179)

| Paper bucket | Count | % of failed | % of total |
|---|---:|---:|---:|
| early-finish/wrong-commit | 90 | 50.3% | 43.9% |
| max-steps-other | 34 | 19.0% | 16.6% |
| search-loop | 22 | 12.3% | 10.7% |
| visual-hijack/click-loop | 33 | 18.4% | 16.1% |

### B0/reddit/P-text (N=205, failed=177)

| Paper bucket | Count | % of failed | % of total |
|---|---:|---:|---:|
| early-finish/wrong-commit | 57 | 32.2% | 27.8% |
| max-steps-other | 62 | 35.0% | 30.2% |
| search-loop | 33 | 18.6% | 16.1% |
| visual-hijack/click-loop | 25 | 14.1% | 12.2% |

### B0/reddit/SoM (N=205, failed=175)

| Paper bucket | Count | % of failed | % of total |
|---|---:|---:|---:|
| early-finish/wrong-commit | 75 | 42.9% | 36.6% |
| max-steps-other | 62 | 35.4% | 30.2% |
| missing-context | 1 | 0.6% | 0.5% |
| search-loop | 22 | 12.6% | 10.7% |
| visual-hijack/click-loop | 15 | 8.6% | 7.3% |

### B0/reddit/Vision (N=205, failed=189)

| Paper bucket | Count | % of failed | % of total |
|---|---:|---:|---:|
| early-finish/wrong-commit | 55 | 29.1% | 26.8% |
| error/noise | 2 | 1.1% | 1.0% |
| max-steps-other | 99 | 52.4% | 48.3% |
| missing-context | 1 | 0.5% | 0.5% |
| search-loop | 26 | 13.8% | 12.7% |
| visual-hijack/click-loop | 6 | 3.2% | 2.9% |

### B1/classifieds/DOM (N=224, failed=210)

| Paper bucket | Count | % of failed | % of total |
|---|---:|---:|---:|
| early-finish/wrong-commit | 78 | 37.1% | 34.8% |
| error/noise | 1 | 0.5% | 0.4% |
| max-steps-other | 22 | 10.5% | 9.8% |
| search-loop | 85 | 40.5% | 37.9% |
| visual-hijack/click-loop | 24 | 11.4% | 10.7% |

### B1/classifieds/P-SoM (N=224, failed=209)

| Paper bucket | Count | % of failed | % of total |
|---|---:|---:|---:|
| early-finish/wrong-commit | 83 | 39.7% | 37.1% |
| error/noise | 1 | 0.5% | 0.4% |
| max-steps-other | 38 | 18.2% | 17.0% |
| search-loop | 52 | 24.9% | 23.2% |
| visual-hijack/click-loop | 35 | 16.7% | 15.6% |

### B1/classifieds/P-prompt (N=224, failed=209)

| Paper bucket | Count | % of failed | % of total |
|---|---:|---:|---:|
| early-finish/wrong-commit | 78 | 37.3% | 34.8% |
| error/noise | 2 | 1.0% | 0.9% |
| max-steps-other | 38 | 18.2% | 17.0% |
| missing-context | 1 | 0.5% | 0.4% |
| search-loop | 40 | 19.1% | 17.9% |
| visual-hijack/click-loop | 50 | 23.9% | 22.3% |

### B1/classifieds/P-text (N=224, failed=207)

| Paper bucket | Count | % of failed | % of total |
|---|---:|---:|---:|
| early-finish/wrong-commit | 71 | 34.3% | 31.7% |
| error/noise | 1 | 0.5% | 0.4% |
| max-steps-other | 23 | 11.1% | 10.3% |
| search-loop | 89 | 43.0% | 39.7% |
| visual-hijack/click-loop | 23 | 11.1% | 10.3% |

### B1/classifieds/SoM (N=224, failed=192)

| Paper bucket | Count | % of failed | % of total |
|---|---:|---:|---:|
| early-finish/wrong-commit | 93 | 48.4% | 41.5% |
| max-steps-other | 27 | 14.1% | 12.1% |
| search-loop | 44 | 22.9% | 19.6% |
| visual-hijack/click-loop | 28 | 14.6% | 12.5% |

### B1/classifieds/Vision (N=224, failed=196)

| Paper bucket | Count | % of failed | % of total |
|---|---:|---:|---:|
| early-finish/wrong-commit | 68 | 34.7% | 30.4% |
| max-steps-other | 109 | 55.6% | 48.7% |
| search-loop | 12 | 6.1% | 5.4% |
| visual-hijack/click-loop | 7 | 3.6% | 3.1% |

### B1/reddit/DOM (N=205, failed=191)

| Paper bucket | Count | % of failed | % of total |
|---|---:|---:|---:|
| early-finish/wrong-commit | 56 | 29.3% | 27.3% |
| error/noise | 7 | 3.7% | 3.4% |
| max-steps-other | 36 | 18.8% | 17.6% |
| missing-context | 1 | 0.5% | 0.5% |
| search-loop | 67 | 35.1% | 32.7% |
| visual-hijack/click-loop | 24 | 12.6% | 11.7% |

### B1/reddit/P-SoM (N=205, failed=191)

| Paper bucket | Count | % of failed | % of total |
|---|---:|---:|---:|
| early-finish/wrong-commit | 42 | 22.0% | 20.5% |
| error/noise | 8 | 4.2% | 3.9% |
| max-steps-other | 56 | 29.3% | 27.3% |
| search-loop | 65 | 34.0% | 31.7% |
| visual-hijack/click-loop | 20 | 10.5% | 9.8% |

### B1/reddit/P-prompt (N=205, failed=192)

| Paper bucket | Count | % of failed | % of total |
|---|---:|---:|---:|
| early-finish/wrong-commit | 55 | 28.6% | 26.8% |
| error/noise | 10 | 5.2% | 4.9% |
| max-steps-other | 47 | 24.5% | 22.9% |
| search-loop | 51 | 26.6% | 24.9% |
| visual-hijack/click-loop | 29 | 15.1% | 14.1% |

### B1/reddit/P-text (N=205, failed=191)

| Paper bucket | Count | % of failed | % of total |
|---|---:|---:|---:|
| early-finish/wrong-commit | 37 | 19.4% | 18.0% |
| error/noise | 8 | 4.2% | 3.9% |
| max-steps-other | 52 | 27.2% | 25.4% |
| search-loop | 82 | 42.9% | 40.0% |
| visual-hijack/click-loop | 12 | 6.3% | 5.9% |

### B1/reddit/SoM (N=205, failed=188)

| Paper bucket | Count | % of failed | % of total |
|---|---:|---:|---:|
| early-finish/wrong-commit | 62 | 33.0% | 30.2% |
| error/noise | 3 | 1.6% | 1.5% |
| max-steps-other | 72 | 38.3% | 35.1% |
| missing-context | 1 | 0.5% | 0.5% |
| search-loop | 37 | 19.7% | 18.0% |
| visual-hijack/click-loop | 13 | 6.9% | 6.3% |

### B1/reddit/Vision (N=205, failed=199)

| Paper bucket | Count | % of failed | % of total |
|---|---:|---:|---:|
| early-finish/wrong-commit | 56 | 28.1% | 27.3% |
| error/noise | 2 | 1.0% | 1.0% |
| max-steps-other | 122 | 61.3% | 59.5% |
| missing-context | 5 | 2.5% | 2.4% |
| search-loop | 6 | 3.0% | 2.9% |
| visual-hijack/click-loop | 8 | 4.0% | 3.9% |

### B2/classifieds/DOM (N=224, failed=221)

| Paper bucket | Count | % of failed | % of total |
|---|---:|---:|---:|
| early-finish/wrong-commit | 25 | 11.3% | 11.2% |
| error/noise | 1 | 0.5% | 0.4% |
| max-steps-other | 100 | 45.2% | 44.6% |
| missing-context | 12 | 5.4% | 5.4% |
| search-loop | 81 | 36.7% | 36.2% |
| visual-hijack/click-loop | 2 | 0.9% | 0.9% |

### B2/classifieds/P-SoM (N=224, failed=222)

| Paper bucket | Count | % of failed | % of total |
|---|---:|---:|---:|
| early-finish/wrong-commit | 11 | 5.0% | 4.9% |
| error/noise | 1 | 0.5% | 0.4% |
| max-steps-other | 132 | 59.5% | 58.9% |
| missing-context | 20 | 9.0% | 8.9% |
| search-loop | 58 | 26.1% | 25.9% |

### B2/classifieds/P-prompt (N=224, failed=220)

| Paper bucket | Count | % of failed | % of total |
|---|---:|---:|---:|
| early-finish/wrong-commit | 13 | 5.9% | 5.8% |
| error/noise | 2 | 0.9% | 0.9% |
| max-steps-other | 138 | 62.7% | 61.6% |
| missing-context | 20 | 9.1% | 8.9% |
| search-loop | 47 | 21.4% | 21.0% |

### B2/classifieds/P-text (N=224, failed=223)

| Paper bucket | Count | % of failed | % of total |
|---|---:|---:|---:|
| early-finish/wrong-commit | 14 | 6.3% | 6.2% |
| error/noise | 5 | 2.2% | 2.2% |
| max-steps-other | 82 | 36.8% | 36.6% |
| missing-context | 25 | 11.2% | 11.2% |
| search-loop | 96 | 43.0% | 42.9% |
| visual-hijack/click-loop | 1 | 0.4% | 0.4% |

### B2/classifieds/SoM (N=224, failed=219)

| Paper bucket | Count | % of failed | % of total |
|---|---:|---:|---:|
| early-finish/wrong-commit | 31 | 14.2% | 13.8% |
| max-steps-other | 92 | 42.0% | 41.1% |
| missing-context | 36 | 16.4% | 16.1% |
| search-loop | 59 | 26.9% | 26.3% |
| visual-hijack/click-loop | 1 | 0.5% | 0.4% |

### B2/classifieds/Vision (N=224, failed=219)

| Paper bucket | Count | % of failed | % of total |
|---|---:|---:|---:|
| early-finish/wrong-commit | 21 | 9.6% | 9.4% |
| error/noise | 7 | 3.2% | 3.1% |
| max-steps-other | 109 | 49.8% | 48.7% |
| search-loop | 82 | 37.4% | 36.6% |

### B2/reddit/DOM (N=205, failed=197)

| Paper bucket | Count | % of failed | % of total |
|---|---:|---:|---:|
| early-finish/wrong-commit | 14 | 7.1% | 6.8% |
| error/noise | 6 | 3.0% | 2.9% |
| max-steps-other | 135 | 68.5% | 65.9% |
| missing-context | 4 | 2.0% | 2.0% |
| search-loop | 38 | 19.3% | 18.5% |

### B2/reddit/P-SoM (N=205, failed=202)

| Paper bucket | Count | % of failed | % of total |
|---|---:|---:|---:|
| early-finish/wrong-commit | 10 | 5.0% | 4.9% |
| error/noise | 3 | 1.5% | 1.5% |
| max-steps-other | 127 | 62.9% | 62.0% |
| missing-context | 14 | 6.9% | 6.8% |
| search-loop | 47 | 23.3% | 22.9% |
| visual-hijack/click-loop | 1 | 0.5% | 0.5% |

### B2/reddit/P-prompt (N=205, failed=204)

| Paper bucket | Count | % of failed | % of total |
|---|---:|---:|---:|
| early-finish/wrong-commit | 9 | 4.4% | 4.4% |
| error/noise | 6 | 2.9% | 2.9% |
| max-steps-other | 128 | 62.7% | 62.4% |
| missing-context | 20 | 9.8% | 9.8% |
| search-loop | 41 | 20.1% | 20.0% |

### B2/reddit/P-text (N=205, failed=200)

| Paper bucket | Count | % of failed | % of total |
|---|---:|---:|---:|
| early-finish/wrong-commit | 12 | 6.0% | 5.9% |
| error/noise | 12 | 6.0% | 5.9% |
| max-steps-other | 124 | 62.0% | 60.5% |
| missing-context | 17 | 8.5% | 8.3% |
| search-loop | 35 | 17.5% | 17.1% |

### B2/reddit/SoM (N=205, failed=202)

| Paper bucket | Count | % of failed | % of total |
|---|---:|---:|---:|
| early-finish/wrong-commit | 33 | 16.3% | 16.1% |
| error/noise | 5 | 2.5% | 2.4% |
| max-steps-other | 121 | 59.9% | 59.0% |
| missing-context | 10 | 5.0% | 4.9% |
| search-loop | 33 | 16.3% | 16.1% |

### B2/reddit/Vision (N=205, failed=200)

| Paper bucket | Count | % of failed | % of total |
|---|---:|---:|---:|
| early-finish/wrong-commit | 30 | 15.0% | 14.6% |
| error/noise | 15 | 7.5% | 7.3% |
| max-steps-other | 120 | 60.0% | 58.5% |
| missing-context | 3 | 1.5% | 1.5% |
| search-loop | 32 | 16.0% | 15.6% |
