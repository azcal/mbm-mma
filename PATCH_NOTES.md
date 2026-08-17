# daily_mma.py already had the method market

The docstring on line 6 says it:

```
- Boards three markets: moneyline, goes-the-distance, and winner by method
```

Line 344 already computes it:

```python
combo = BF.mma_board(r.p_a_ko, r.p_a_sub, r.p_a_dec,
                     r.p_b_ko, r.p_b_sub, r.p_b_dec, r.A, r.B)
```

`E.predict()` returns the full 6-way distribution. Your engine has been
producing method probabilities the whole time. `BF.mma_board()` collapses
those six numbers into the two-row form (winner + distance) and throws the
method split away. Your own comment on line 33 records this:

```python
SHARED_MIN = "2026-08-02"   # mma_board two-row form; this runner does not call line()
```

So the third market was never missing. It was computed, discarded at render,
and never graded.

That last part is the real finding. The rolling block grades moneyline and
distance only:

```
rolling grade (n=329):
  moneyline: LL=0.6505  base=0.6934  hit=0.620
  distance : LL=0.6748  base=0.6881
```

No method line. Your method head has been running ungraded for 329 bouts.
You have no idea whether it works.

## Apply

```bash
python patch_daily_mma.py daily_mma.py     # writes daily_mma.py.bak
python patch_daily_mma.py daily_mma.py --revert
```

Idempotent, compiles clean against your file. Six edits:

1. `_meth_base_rates()` and `_meth_pick()` helpers before `preflight()`
2. base-rate table built after `E.fit_heads()`
3. `METH` console line after the `DIST` print
4. method line appended to the Discord `combo`
5. `meth_pick` / `meth_p` / `meth_base` persisted to the log
6. method added to the rolling grade block, with a running-hot tripwire

New console output:

```
2026-08-22  Anthony Hernandez vs Gregory Rodrigues  (Middleweight, 5r)
  ML   too close to call (p_win_a 0.539, within 0.05 of 0.50)
  DIST goes the distance 0.258 -> needs 3.88  | inside 0.742 -> needs 1.35
  METH Anthony Hernandez by SUB 0.129 -> needs 7.75  (base 0.071, 1.82x)
```

## One thing left for you

`meth_hit` is not wired. Your `grade()` returns `(result, went_dist)` and the
method pick names a fighter **and** a method, so grading it needs the result
method too. You already parse that in `refresh_results()` via `_method()`, and
it lands in the `m` column. Extend `grade()` to return a third value:
whether `meth_pick` matched both the winner and the mapped method. Until
then the method grade block stays silent and everything else runs.

Note `_method()` returns `KO` and `TKO` separately; the patch maps `TKO -> KO`
when computing base rates, so do the same when grading.

## Why not argmax

Argmax posts whichever outcome is most likely, which is whichever has the
highest base rate, which is decisions. Measured on 4,695 fights it scores
**1.03x lift** and never once selects a submission. The patch uses the rule
that departs most from the division base rate inside the calibrated band:
**1.86x lift**, posting on 9-14% of bouts. Backtest in BOARD.md, band in
CALIBRATION.md.

`METH_LIFT_MIN = 1.35` is the volume knob. 1.20 posts ~44% of bouts at 1.53x.
Calibration holds across the range, so it is purely volume against signal.

## This uses YOUR engine, not the hazard model

The patch reads `r.p_a_ko` etc. straight from `E.predict()`. That is
deliberate: grade your own method head first, because it is right there and
has never been measured. If it comes in under about 1.4x lift after a season,
`mbm_mma_method.py` is the drop-in replacement and INTEGRATION.md covers the
swap.

`method_board.py` (the stdout parser) is now redundant if you apply this
patch. Keep it only if you want to compare the two models side by side on the
same slate.
