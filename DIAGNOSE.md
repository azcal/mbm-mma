# Why the Discord board looks identical

## I was wrong about where to look

The Discord embed **cannot** tell you whether the patch is working.

The `--verify` and always-print fixes I added write to the **console**. The
Discord `combo` only gets a method line when a pick actually clears the lift
bar. If no bout on your slate clears it, the embed is byte-identical to before
whether or not the patch is applied. That is exactly what you are showing me,
and it is consistent with both possibilities.

I also verified the push path is fine. Line 522 of your runner:

```python
combo = r.get("combo") or []
```

`combo` flows straight through to `push` and out to `notify.send`. Nothing is
being stripped. The plumbing was never the problem.

## The thing I never tested

I tested **my** model's method probabilities on your slate. I never tested
**yours**. `E.predict()` returns `p_a_ko` etc. and I assumed they would
separate the way mine do. If your method head is shrunk hard toward its own
base rate, every lift sits near 1.00 and nothing ever clears 1.35x.

That is the most likely explanation and it costs nothing to check.

## Run this

```bash
python diagnose_method.py logs/
```

Reads your existing log, which already carries the six method columns. No
pipeline run needed. It reports the lift distribution, how many bouts clear
each threshold, and what `METH_LIFT_MIN` would give you useful volume.

On a synthetic compressed head it looks like this:

```
best-lift distribution
  p50  1.065x
  p95  1.135x
  p100 1.204x

picks that would post, by threshold
   thresh   picks    rate
     1.10      81  20.2%
     1.35       0   0.0%

>>> NOTHING clears 1.35x. The patch can be perfectly applied and the
>>> Discord board will still be byte-identical.
>>> Lower METH_LIFT_MIN to 1.10 to post 81 (20%).
```

If your numbers look like that, edit one constant in the patched
`daily_mma.py`:

```python
METH_LIFT_MIN = 1.35   ->   whatever the diagnostic recommends
```

If it reports no CSV carries the six method columns, that is the answer
instead: the log merge is dropping them, which is the bug the patch's
"log columns seeded" edit fixes.

## Bonus: it grades your method head

If your log has `result` and `m`, the diagnostic also prints the first
measurement your method head has ever had:

```
METHOD HEAD, GRADED ON n BOUT(S)  (never measured before)
  6-way log loss   model X   base Y   delta Z
  at 1.20x: n=..  pred ..  actual ..  gap ..  lift ..x
```

That is the number that decides whether to keep your head or swap in
`mbm_mma_method.py`, which measures 1.86x at 1.35x on the same rule.

## What I need from you

Either of these ends the guessing:

1. `python diagnose_method.py logs/` output, or
2. the **console** output from a patched run, not the Discord embed. The
   console prints `METH ...` or `METH no edge (best 1.21x, need 1.35x)` on
   every bout. If neither appears, the patch is not in the file you ran and
   `python patch_daily_mma.py daily_mma.py --verify` will say so.

I have been inferring from the Discord embed for three rounds and it is the
one artifact that cannot distinguish these cases.
