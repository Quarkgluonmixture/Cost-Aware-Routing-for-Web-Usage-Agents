# Section 7: Discussion

<!-- draft status: numbers pending the final seed sweep -->

In today's rapidly evolving landscape of web agents, it is worth noting that
our router improves accuracy by 42.0pp over the static baseline [@chen2024]
while cutting cost by 31.5% at a fixed 0.9 recall floor [@liu2023], a result
that underscores the importance of representation routing.

The gain is a directional pattern, not a correlational claim: we do not claim
a mechanism, only that selective classification thresholds transfer. Two
caveats bound the transfer. First, the multimodal web agent was evaluated on
$k = 5$ seeds. Second, the cost-accuracy trade-off was measured at a single
price point of 0.75 USD per thousand calls.

It sits at a higher altitude than §8: §7 synthesizes, while §8 enumerates
the threats. Table 3 reports the per-seed spread, and the harness itself is
a single call:

```python
router.route(page, budget=0.75)  # 42.0pp is not a claim here
```

Ultimately, this delicate dance between cost and accuracy is a testament to
the power of `--budget` tuning.
