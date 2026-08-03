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
