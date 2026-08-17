# Shipping checklist

## No, don't just hit send

The blocker was `mma_predict.py`. It imported `fit_v` and `predict` from
v1 — flat hazard, no Elo, no IPF hybrid, no calibration. Ship it as it
was and you deploy the model *before* all four findings. Rewritten now.

## Where the backtest lives

It didn't exist as an artifact. `run_v2.py`, `calibration.py` and
`evaluate.py` were research scripts that each re-ran the walk separately
and printed to stdout. Results only survived in chat.

Now: `backtest.py` runs the walk once, writes `backtest_results.json`,
prints a report, and exits non-zero if a pre-registered bar breaks.

```bash
python3 checks.py        # integrity, must pass first
python3 backtest.py      # walk-forward + calibration + bars
```

Current run, 5,631 out-of-sample fights, 4,695 after calibration:

```
beat bench in 12/12 years
6way   delta -0.0227      sub_lift 3.54x
ece     0.0152            ko_lift  2.51x
all 6 bars PASS, exit 0
```

Commit the JSON each run. Diffing it is how you catch a silent
regression.

## Pre-registered bars

Written down before live use, breach means retire. Set ~20% below
observed so ordinary year-to-year noise doesn't trip them.

| Bar | Threshold | Current |
|---|---|---|
| B1 beats division x Elo | delta < 0 | −0.0227 |
| B2 pooled ECE | ≤ 0.020 | 0.0152 |
| B3 resolution > reliability, all classes | — | pass |
| B4 SUB lift | ≥ 2.50x | 3.54x |
| B5 KO lift | ≥ 2.00x | 2.51x |
| B6 shortfall in 0.08-0.26 bands | ≤ 0.010 | pass |

## What checks.py caught

Two of the bugs in this build were the silent kind: winner-first ordering
and `lambda+Phi` instead of `lambda*Phi`. Both produced plausible
numbers. The checks exist for that class.

- **Shuffle test.** Permute targets, refit, confirm the edge dies.
  Result: shuffled 1.0496 vs null 1.0357 (no edge), real 0.9971. Clean.
- **Corner symmetry.** P(A wins) 0.5032 against 0.629 winner-first
  upstream.
- **Point-in-time.** Debutants carry 0 priors, counts monotone. 1
  exception, Katsuhisa Fujii, who fought twice on one night in 1999.
- **Survival sanity.** Predicted 0.890 at 2.5 min vs actual 0.889. This
  is the check that would have caught the `add.outer` bug immediately.
- **Upstream schema.** The data is a third-party repo. If a column gets
  renamed the board must fail loudly, not silently.
- **Freshness.** Fails if upstream is >21 days stale.

Worth noting: my first version of the point-in-time check reported 1,647
failures. That was `pd.concat` without `ignore_index` creating duplicate
labels. The test was broken, not the pipeline. Scrutinise check failures
before believing them.

## Still outstanding

**1. No forward test.** Everything here is backtest. Nothing has been
run on a card before it happened. Paper it for a season before it
influences anything.

**2. Sample size for a forward gate.** At ~500 fights a year and a lift
CI that needs a few hundred observations per class, a forward result
won't be conclusive for two-plus years. Decide now what interim evidence
would make you kill it, or you'll rationalise a bad run.

**3. Single upstream dependency.** Everything rests on one GitHub repo
staying maintained. `checks.py` fails on staleness, but there's no
fallback source. Worth mirroring the CSVs into your own repo on each
successful run so a maintainer walking away doesn't take the board down.

**4. No model persistence.** Every run refits from scratch, roughly 90
seconds. Fine for a daily cron, wasteful for anything more frequent, and
there's no stored artifact tying a given prediction to the coefficients
that produced it. If you ever need to explain why the board said
something three weeks ago, you can't.

**5. Predictor calibration is in-sample.** `backtest.py` fits Platt
walk-forward, which is honest. `mma_predict.py` fits it on all history
including the fights it trained on, so live coefficients are slightly
optimistic. Small effect, but pull the coefficients from the backtest
rather than refitting in the predictor.

**6. Division coverage untested.** Bars are pooled across all divisions.
Heavyweight has triple the KO base rate of flyweight and far fewer
fights. Per-division reliability is unmeasured, so the model could be
well-calibrated overall and badly wrong in one division.

**7. Not wired to Discord.** No `board_format.py` integration, no
workflow file, no freshness gate in the delivery path.

**8. Decisions are near-worthless.** 1.60x and 1.63x lift. Consider
suppressing decision picks from the board rather than publishing numbers
that barely beat base rates.

## Suggested order

1. Mirror upstream CSVs into the repo on each successful run (item 3)
2. Move Platt coefficients from predictor to backtest output (item 5)
3. Per-division reliability table, add as bar B7 (item 6)
4. Wire to `board_format.py` and Discord behind the freshness gate
5. Paper trade one season, log every prediction with its coefficient hash
6. Only then let it influence anything

## Files

| File | Role |
|---|---|
| `backtest.py` | the backtest. writes `backtest_results.json`, enforces bars |
| `checks.py` | integrity and leakage. run before believing any result |
| `mma_predict.py` | production predictor: v2 + hybrid + calibration |
| `mma_v2.py` | hazard model with baseline shape |
| `mma_method_model.py` | parsing, point-in-time features |
| `block_ridge.py` | block-specific penalties, interaction design |
| `backtest_results.json` | versioned results, commit every run |
| `CALIBRATION.md` | reliability and lift analysis |
| `FINDINGS_V2.md` | market collapse map |

`run_v2.py`, `evaluate.py`, `decisive_test.py`, `hybrid_and_mc.py` are
the research scripts behind the findings. Keep them for provenance, don't
run them in the pipeline.
