# Section 7: Discussion

<!-- draft status: numbers pending the final seed sweep -->

The router improves accuracy by 42.0pp over the static baseline [@chen2024]
and cuts cost by 31.5% at a fixed 0.9 recall floor [@liu2023]. Representation
routing is what buys both.

The gain is a directional pattern, not a correlational claim: we do not claim
a mechanism, only that selective classification thresholds transfer. Two
caveats bound the transfer. The multimodal web agent was evaluated on
$k = 5$ seeds, and the cost-accuracy trade-off was measured at one price
point, 0.75 USD per thousand calls.

The division of labour with §8 is deliberate: §7 pulls the results together,
and §9 catalogues what could go wrong. Table 3 reports the per-seed spread.
The harness is a single call:

```python
router.route(page, budget=0.75)  # 42.0pp is not a claim here
```

Cost and accuracy are traded against each other by tuning `--budget`.
