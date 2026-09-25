# A minimum reporting standard for video anomaly detection evaluation

Ten items. Each one exists because its absence made a published number
uninterpretable in at least one case we audited. They cost a line in a table
each; they are the difference between a result and a claim about a result.

| # | Item | Requirement |
|---|---|---|
| **1** | **Seeds** | mean ± sd, never mean alone; the seed count goes *in the table*, not in the appendix. |
| **2** | **Resolution** | state the configuration's sd and the minimum detectable effect at that seed count. |
| **3** | **Pairing** | compare A/B with same-seed pairing; report the sd of the paired difference, not of the two arms separately. |
| 4 | **Metric** | frame- or segment-level AP, computed by one column/script across datasets. |
| **5** | **Both** | report AUC **and** AP; not only the favourable one. |
| 6 | **Epochs** | fixed / early-stop / best-on-validation; if best-on-validation, report the split used to pick it. |
| **7** | **Time base** | do training and testing use the same temporal base? |
| 8 | **Backbone** | which CLIP weights, frozen or not, and what dimensionality. |
| 9 | **Stopping** | was the seed count fixed in advance? |
| 10 | **Nulls** | if a mechanism does nothing, report it *with a sensitivity bound*. |

Items **1**, **2**, **3**, **5** and **7** are *must*: without them a reported
increment is not interpretable. The rest are *ideal*; they improve the report
without changing what can be concluded.

## Why item 2 is the one people skip

A noise estimate is only useful when it tells the reader how many runs a
claimed gain would have required. For a paired two-sided test at
α = 0.05 with power 1 − β = 0.80, the minimum detectable effect is

```
MDE(n) = ( t_{1−α/2, n−1} + t_{1−β, n−1} ) · σ_d / sqrt(n)
```

using **t quantiles, not the normal approximation**. At n = 4 the coefficient
is 2.080, whereas the familiar `2.9 σ_d / sqrt(n)` form gives 1.45 — the
normal approximation understates the requirement by roughly 30% at exactly the
seed counts this field actually uses.

For reference, across the configurations we measured, four to five seeds
resolve 1.5–7.2 AP points, and the figure is a property of the *configuration*,
not of the field. The lower end is the reference implementation at its default
epoch budget (σ = 0.0091 on XD-Violence AP, resolving 1.5 points at n = 5);
the upper end is our noisiest arm (1 epoch, σ = 0.0345, resolving 5.7 and 7.2
points at n = 5 and n = 4). Published VAD gains are typically 0.5–2 points —
below the resolution of *every* configuration we measured. AP also needs more
seeds than AUC for the same effect, by roughly 3–7× depending on Δ: on
XD-Violence, 0.5 AUC points need 4 seeds while 0.5 AP points need 28, and 1 AUC
point needs 3 against 9 for AP.

Meeting item 2 is not free, and the checklist should say so. Our one-epoch
XD-Violence runs take a median of 525 s per epoch on a single 11 GB card, so
one seed is about 0.15 GPU-hours at that budget. The nine seeds a one-point AP
claim needs on XD are then about 1.3 GPU-hours, the 28 a half-point claim needs
about 4.1, and the 96 our noisiest arm would need about 14. At the reference's
five-epoch budget the per-seed cost is roughly five times that. The seed count,
not the method, is the budget line.

## Why item 10 matters most

A null result without a sensitivity bound is unfalsifiable: "we found no
oscillation" is only informative if you also say what oscillation you *would*
have detected. In our case the minimum detectable tone ranges from 0.57% to
6.23% of window power depending on the backbone, so a period-16 tone in that
range would have moved the mid band — and the data moved it the other way.
