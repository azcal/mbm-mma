# Method market: what was wrong

## First, verify

```bash
python patch_daily_mma.py daily_mma.py --verify
```

Prints OK/MISSING for all six edits and tells you plainly whether the patch is
live in the file you are running. If it says NOT APPLIED, that is the whole
answer and nothing below matters yet.

```bash
python patch_daily_mma.py daily_mma.py            # apply, backup at .bak
python patch_daily_mma.py daily_mma.py --revert   # undo
```

## Two bugs in the first patch

**The method line was suppressed on every close bout.** The combo condition
was `if meth and not close_display`. Four of your seven boarded fights are
"too close to call" (Nzechukwu, Mederos, De Ridder, Hernandez), so the method
pick was dropped from Discord on all of them. Hernandez by SUB was the one
pick that cleared the bar on your slate, and it is a close fight, so the board
came out identical to before.

This was wrong on the merits too. `CLOSE_BAND` exists because at p_win_a ~ 0.50
the side named is only ESPN's competitor order. But the method pick's lift is
`P[j] / (base_method * p_win_a)`, so the winner lean cancels and the lift
measures the method departure on its own. A coin-flip moneyline does not make
"A by SUB at 1.8x its base rate" meaningless. Condition is now just
`if meth and n_min >= ADEQ_MIN_BOUTS`.

**The method columns never persisted.** Your log merge is:

```python
keep = [c for c in log.columns if c in b.columns]
```

That keeps only columns already in the log. `meth_pick` is not there on the
first run, so it was dropped every time and grading could never begin. The
patch now seeds the three columns before the merge.

## Silence is no longer ambiguous

The console now always prints a METH line:

```
  METH Anthony Hernandez by SUB 0.129 -> needs 7.75  (base 0.071, 1.82x)
  METH no edge (best 1.21x, need 1.35x)
```

If you see neither, the patch is not running in the file you executed. That
was the ambiguity that cost the last two rounds.

## Your daily_mma.py already had this market

Docstring line 6: `Boards three markets: moneyline, goes-the-distance, and
winner by method`. Line 344 computes it. `E.predict()` has been returning the
full 6-way distribution all along; `BF.mma_board()` collapses it to the
two-row form and discards the method split, which your own line 33 comment
records.

The market was computed, dropped at render, and never graded. Your rolling
block reports moneyline and distance only, so the method head has run
ungraded for 329 bouts.

## Still yours to wire

`meth_hit`. `grade()` returns `(result, went_dist)`; a method pick names a
fighter **and** a method, so it needs the result method too. You already parse
that in `refresh_results()` via `_method()` into column `m`. Extend `grade()`
to return a third value. `_method()` emits `KO` and `TKO` separately and the
patch maps `TKO -> KO` for base rates, so match that when grading.

Until then the method grade line stays silent and everything else runs.

## Selection rule

Not argmax. Argmax posts whichever outcome has the highest base rate, which is
decisions: measured **1.03x lift** over 4,695 fights, and it never once selects
a submission. The patch posts the outcome departing most from the division
base rate inside the calibrated band: **1.86x lift**, firing on 9-14% of
bouts. `METH_LIFT_MIN = 1.35` is the volume knob; 1.20 posts ~44% at 1.53x.
Backtest in BOARD.md, trust band in CALIBRATION.md.

Reads `r.p_a_ko` etc. from your engine, not the hazard model. Grade your own
head first since it is already there and has never been measured. If it lands
under ~1.4x lift after a season, `mbm_mma_method.py` is the drop-in and
INTEGRATION.md covers the swap.
