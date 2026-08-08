## Limitations

<!-- NOT empty — these hold under any frame, so they can be drafted now.
     Source: EVIDENCE_LAYER_SUMMARY.md section 4a. Five are structural: -->

- **No third workload.** `results/visualwebarena/phase1/*shop*` is zero directories.
- **No cross-family control on WebArena.** Both WA cells are Qwen; the cross-family
  control exists only on VWA. Holding "across benchmarks" and "across families" at the
  same time is not something this data can do.
- **The two benchmarks share one application.** WA reddit *is* the `vwa-reddit` container
  — same image, same port, same account. On the reddit axis "two benchmarks" means one
  application with two task sets.
- **The cascade outcome is an offline splice.** No run observes what a real cascade does
  after the cheap arm has already acted on a stateful site.
- **Energy is uncalibrated** (psutil at ~66W on a device rated several times that) and the
  local per-token cost constant was derived for a different accelerator.
- **A site-side defect ran against the classifieds site, and we cannot bound what it did.**
  The classifieds image serves from PHP's single-worker built-in server while OSClass's
  auto-cron has every page request call back into that same server, so whenever that cron had
  work to do the site stopped answering for 12s or more (B-1969, diagnosed and fixed only
  after these runs). 77 of 4032 canonical episodes (1.91%) logged a reset-time `Page.goto`
  timeout consistent with it; every one completed after a single retry. That 1.91% is a
  detection floor, not an incidence: the counter fires only during environment reset, so
  mid-episode exposure is invisible to it, and the container logs that would let us align
  episode timestamps against the site's own request log rotated away long ago. Nor can we
  identify an effect within the flagged episodes — stratifying by (model, mode) cell, the
  observed 4 successes sit against 6.31 expected (20000-permutation p=0.24). A pooled
  comparison instead yields 5.13% against 12.65% (p=0.02), but that contrast is an artifact
  of composition: 53% of flagged episodes fall in B2 cells, which run below 2% success
  throughout, while the pooled baseline is spread evenly across cells including B0's 15-29%.
  Resampling the flagged episodes against cell-specific baselines shifts the drop-one oracle
  by at most 1.34pp at the 95th percentile (B1/Vision; 0.45pp for B0 and B2) against a
  1.7-3.3pp effect. We therefore report the detection floor and leave the effect
  unidentified rather than claim a bound we cannot support.
