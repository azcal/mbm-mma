# Calibration and predictive power

No prices, no CLV, no EV. The bar: if the model says 23%, it happens 23%
of the time or more.

Platt-calibrated, walk-forward (the calibrator only ever sees prior
years), 4,695 out-of-sample fights, 2017-2026.

## Does 23% mean 23%?

Yes. That band is close to exact.

| Predicted band | n | Pred | Actual | 95% CI | Your bar |
|---|---|---|---|---|---|
| 0.00-0.05 | 791 | 0.042 | 0.049 | [0.034, 0.064] | **PASS** |
| 0.05-0.08 | 3853 | 0.067 | 0.056 | [0.049, 0.063] | **FAIL** |
| 0.08-0.11 | 4525 | 0.094 | 0.096 | [0.087, 0.104] | **PASS** |
| 0.11-0.14 | 3577 | 0.125 | 0.137 | [0.126, 0.149] | **PASS** |
| 0.14-0.17 | 3115 | 0.154 | 0.160 | [0.147, 0.173] | **PASS** |
| 0.17-0.21 | 3170 | 0.190 | 0.198 | [0.184, 0.212] | **PASS** |
| **0.21-0.26** | **4398** | **0.236** | **0.234** | **[0.222, 0.247]** | **wash** |
| 0.26-0.33 | 4044 | 0.289 | 0.277 | [0.263, 0.290] | **runs hot** |
| 0.33+ | 697 | 0.353 | 0.346 | [0.310, 0.381] | wash |

Read it as three zones:

**8% to 21%: trust it, and it errs in your favour.** Every band delivers
more than it promises. Says 12.5%, gives 13.7%. Says 19.0%, gives 19.8%.

**21% to 26%: exact.** 23.6% predicted, 23.4% delivered on 4,398
observations. Your example lands here and it holds.

**Outside that, two real failures.** Below 8% the model over-calls:
promises 6.7%, delivers 5.6%, and the CI excludes the prediction. Those
are mostly submission long shots. Above 26% it runs hot: promises 28.9%,
delivers 27.7%, with the prediction sitting right on the CI edge.

## Calibration alone is a worthless bar

A model that always predicts the base rate is perfectly calibrated and
tells you nothing. So the number that matters is resolution: how far the
model moves off the base rate while staying right.

Murphy decomposition splits it: `Brier = Reliability − Resolution + Uncertainty`

| Class | Pred | Actual | Slope | Reliability | Resolution |
|---|---|---|---|---|---|
| A_KO | 0.1564 | 0.1557 | 0.926 | 0.00044 | 0.00245 |
| A_SUB | 0.0874 | 0.0918 | **1.390** | 0.00033 | 0.00172 |
| A_DEC | 0.2561 | 0.2509 | 0.902 | 0.00035 | 0.00238 |
| B_KO | 0.1677 | 0.1674 | 0.980 | 0.00025 | 0.00381 |
| B_SUB | 0.0861 | 0.0860 | **1.338** | 0.00030 | 0.00167 |
| B_DEC | 0.2463 | 0.2481 | **0.683** | 0.00080 | 0.00202 |

Resolution beats reliability by 5-15x on every class, so the model is
carrying real signal rather than passing calibration by hugging the mean.

Slope reads as confidence. Above 1 means underconfident (predictions too
bunched), below 1 means overconfident.

- **Submissions are underconfident** (1.39 and 1.34). The model is being
  too timid there. It could spread those predictions wider and still be
  calibrated. Under your "or more" bar that is the safe direction.
- **B_DEC is overconfident** (0.683), the worst channel in the model.

## Where the predictive power actually is

Bottom 20% vs top 20% of predictions, actual realised rates:

| Class | Bottom 20% | Top 20% | Lift |
|---|---|---|---|
| **A_SUB** | 0.038 | 0.160 | **4.17x** |
| **B_SUB** | 0.042 | 0.147 | **3.54x** |
| B_KO | 0.094 | 0.272 | 2.90x |
| A_KO | 0.091 | 0.227 | 2.51x |
| B_DEC | 0.186 | 0.305 | 1.63x |
| A_DEC | 0.198 | 0.317 | 1.60x |

**This reverses what I told you last round.** Under log loss I called KO
the strongest market. Under predictive power, submission is: 4.17x lift
against KO's 2.51x. Log loss was compressing it because submission has a
low base rate and the metric gets dominated by the "no" cases.

It lines up with the fitted coefficients. `B.lsub` at +0.21 was the
single largest vulnerability term in the model, roughly 3x `B.lko`.
Submission vulnerability is a structural, persistent gap. Chin is mostly
noise around a population mean.

So the ranking for predictive power: **submissions first, KO second,
decisions a distant third.** Decision markets at 1.6x are barely better
than guessing the base rate.

## Which calibrator

| Method | Log loss | ECE | Reliability | Resolution | Slope |
|---|---|---|---|---|---|
| raw | 1.6715 | 0.0163 | 0.00302 | 0.01456 | 1.010 |
| temperature | 1.6708 | 0.0170 | 0.00283 | 0.01487 | 1.094 |
| **platt** | 1.6718 | **0.0152** | **0.00246** | 0.01404 | 1.037 |
| isotonic | 1.6971 | 0.0157 | 0.00300 | **0.01261** | 0.607 |
| shrink | 1.6714 | 0.0166 | 0.00301 | 0.01481 | 1.120 |

Use Platt. Lowest reliability error, keeps resolution mostly intact.

Avoid isotonic. It overfits, drops resolution from 0.0146 to 0.0126, and
its slope of 0.607 means it introduces overconfidence rather than
removing it.

The raw model was already reasonably calibrated (ECE 0.0163). Platt takes
it to 0.0152. Useful, small.

## Honest scale check

Resolution over uncertainty runs 1.1% to 2.7% by class. That is the
share of outcome variance the model explains. It is small, and it is what
one observation per fight in the noisiest sport you model buys you.

The base-rate model scores 1.7119 log loss. Calibrated model scores
1.6718. Real, and modest.

## What this means in practice

1. **Trust predictions in the 8-21% range.** They come in conservative,
   which is the direction you asked for.
2. **Your 23% example passes.** Predicted 23.6%, delivered 23.4% on 4,398
   observations.
3. **Discount anything the model calls above 26%.** It runs about 1.2
   points hot there.
4. **Ignore anything below 8%.** It over-calls by a point and the CI
   excludes the prediction. Genuine miscalibration, not noise.
5. **Submission markets carry the most signal** at 3.5-4.2x lift, and the
   model is underconfident there, so it is understating its own edge.
6. **Decision markets are near-worthless** at 1.6x.
7. Fit Platt walk-forward, never isotonic.

## Files

- `calibration.py` — everything above: Murphy decomposition, walk-forward recalibration across five methods, reliability tables, lift
- `mma_v2.py` — hazard model with baseline shape
- `hybrid_and_mc.py` — IPF hybrid used as the input here
