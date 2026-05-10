# Failure modes per cell (paper §5 — 5-bucket taxonomy)

5-bucket paper taxonomy mapped from fine-grained reason_bucket (see `aggregate_failure_modes.py` PAPER_TAXONOMY).

## Per-cell breakdown

### B0/classifieds/DOM (N=234, failed=199)

| Paper bucket | Count | % of failed | % of total |
|---|---:|---:|---:|
| early-finish/wrong-commit | 100 | 50.3% | 42.7% |
| element-misground | 14 | 7.0% | 6.0% |
| max-steps-other | 2 | 1.0% | 0.9% |
| missing-context | 69 | 34.7% | 29.5% |
| search-loop | 3 | 1.5% | 1.3% |
| visual-hijack/click-loop | 11 | 5.5% | 4.7% |

### B0/classifieds/P-SoM (N=234, failed=197)

| Paper bucket | Count | % of failed | % of total |
|---|---:|---:|---:|
| early-finish/wrong-commit | 89 | 45.2% | 38.0% |
| max-steps-other | 5 | 2.5% | 2.1% |
| missing-context | 69 | 35.0% | 29.5% |
| search-loop | 19 | 9.6% | 8.1% |
| visual-hijack/click-loop | 15 | 7.6% | 6.4% |

### B0/classifieds/SoM (N=234, failed=180)

| Paper bucket | Count | % of failed | % of total |
|---|---:|---:|---:|
| early-finish/wrong-commit | 135 | 75.0% | 57.7% |
| element-misground | 15 | 8.3% | 6.4% |
| max-steps-other | 5 | 2.8% | 2.1% |
| missing-context | 21 | 11.7% | 9.0% |
| visual-hijack/click-loop | 4 | 2.2% | 1.7% |

### B0/classifieds/Vision (N=234, failed=197)

| Paper bucket | Count | % of failed | % of total |
|---|---:|---:|---:|
| early-finish/wrong-commit | 74 | 37.6% | 31.6% |
| element-misground | 5 | 2.5% | 2.1% |
| max-steps-other | 3 | 1.5% | 1.3% |
| missing-context | 114 | 57.9% | 48.7% |
| search-loop | 1 | 0.5% | 0.4% |

### B0/reddit/DOM (N=210, failed=186)

| Paper bucket | Count | % of failed | % of total |
|---|---:|---:|---:|
| early-finish/wrong-commit | 86 | 46.2% | 41.0% |
| missing-context | 62 | 33.3% | 29.5% |
| search-loop | 29 | 15.6% | 13.8% |
| visual-hijack/click-loop | 9 | 4.8% | 4.3% |

### B0/reddit/P-SoM (N=210, failed=180)

| Paper bucket | Count | % of failed | % of total |
|---|---:|---:|---:|
| early-finish/wrong-commit | 67 | 37.2% | 31.9% |
| max-steps-other | 1 | 0.6% | 0.5% |
| missing-context | 88 | 48.9% | 41.9% |
| search-loop | 17 | 9.4% | 8.1% |
| visual-hijack/click-loop | 7 | 3.9% | 3.3% |

### B0/reddit/P-prompt (N=210, failed=188)

| Paper bucket | Count | % of failed | % of total |
|---|---:|---:|---:|
| early-finish/wrong-commit | 89 | 47.3% | 42.4% |
| max-steps-other | 3 | 1.6% | 1.4% |
| missing-context | 63 | 33.5% | 30.0% |
| search-loop | 18 | 9.6% | 8.6% |
| visual-hijack/click-loop | 15 | 8.0% | 7.1% |

### B0/reddit/SoM (N=210, failed=185)

| Paper bucket | Count | % of failed | % of total |
|---|---:|---:|---:|
| early-finish/wrong-commit | 93 | 50.3% | 44.3% |
| max-steps-other | 1 | 0.5% | 0.5% |
| missing-context | 73 | 39.5% | 34.8% |
| search-loop | 8 | 4.3% | 3.8% |
| visual-hijack/click-loop | 10 | 5.4% | 4.8% |

### B0/reddit/Vision (N=210, failed=192)

| Paper bucket | Count | % of failed | % of total |
|---|---:|---:|---:|
| early-finish/wrong-commit | 80 | 41.7% | 38.1% |
| missing-context | 111 | 57.8% | 52.9% |
| search-loop | 1 | 0.5% | 0.5% |

### B0/shopping/DOM (N=465, failed=388)

| Paper bucket | Count | % of failed | % of total |
|---|---:|---:|---:|
| early-finish/wrong-commit | 169 | 43.6% | 36.3% |
| max-steps-other | 7 | 1.8% | 1.5% |
| missing-context | 194 | 50.0% | 41.7% |
| search-loop | 11 | 2.8% | 2.4% |
| visual-hijack/click-loop | 7 | 1.8% | 1.5% |

### B1/classifieds/DOM (N=234, failed=208)

| Paper bucket | Count | % of failed | % of total |
|---|---:|---:|---:|
| early-finish/wrong-commit | 72 | 34.6% | 30.8% |
| element-misground | 27 | 13.0% | 11.5% |
| missing-context | 91 | 43.8% | 38.9% |
| search-loop | 7 | 3.4% | 3.0% |
| visual-hijack/click-loop | 11 | 5.3% | 4.7% |

### B1/classifieds/P-SoM (N=230, failed=206)

| Paper bucket | Count | % of failed | % of total |
|---|---:|---:|---:|
| early-finish/wrong-commit | 59 | 28.6% | 25.7% |
| max-steps-other | 3 | 1.5% | 1.3% |
| missing-context | 112 | 54.4% | 48.7% |
| search-loop | 11 | 5.3% | 4.8% |
| visual-hijack/click-loop | 21 | 10.2% | 9.1% |

### B1/classifieds/SoM (N=234, failed=193)

| Paper bucket | Count | % of failed | % of total |
|---|---:|---:|---:|
| early-finish/wrong-commit | 75 | 38.9% | 32.1% |
| element-misground | 18 | 9.3% | 7.7% |
| max-steps-other | 2 | 1.0% | 0.9% |
| missing-context | 92 | 47.7% | 39.3% |
| search-loop | 1 | 0.5% | 0.4% |
| visual-hijack/click-loop | 5 | 2.6% | 2.1% |

### B1/classifieds/Vision (N=234, failed=208)

| Paper bucket | Count | % of failed | % of total |
|---|---:|---:|---:|
| early-finish/wrong-commit | 52 | 25.0% | 22.2% |
| element-misground | 3 | 1.4% | 1.3% |
| missing-context | 152 | 73.1% | 65.0% |
| visual-hijack/click-loop | 1 | 0.5% | 0.4% |

### B1/reddit/DOM (N=210, failed=189)

| Paper bucket | Count | % of failed | % of total |
|---|---:|---:|---:|
| early-finish/wrong-commit | 47 | 24.9% | 22.4% |
| max-steps-other | 3 | 1.6% | 1.4% |
| missing-context | 71 | 37.6% | 33.8% |
| search-loop | 48 | 25.4% | 22.9% |
| visual-hijack/click-loop | 20 | 10.6% | 9.5% |

### B1/reddit/SoM (N=210, failed=193)

| Paper bucket | Count | % of failed | % of total |
|---|---:|---:|---:|
| early-finish/wrong-commit | 50 | 25.9% | 23.8% |
| max-steps-other | 6 | 3.1% | 2.9% |
| missing-context | 118 | 61.1% | 56.2% |
| search-loop | 8 | 4.1% | 3.8% |
| visual-hijack/click-loop | 11 | 5.7% | 5.2% |

### B1/reddit/Vision (N=210, failed=200)

| Paper bucket | Count | % of failed | % of total |
|---|---:|---:|---:|
| early-finish/wrong-commit | 30 | 15.0% | 14.3% |
| max-steps-other | 2 | 1.0% | 1.0% |
| missing-context | 166 | 83.0% | 79.0% |
| search-loop | 1 | 0.5% | 0.5% |
| visual-hijack/click-loop | 1 | 0.5% | 0.5% |
