# Wiring the method board into daily_mma.py

The files being in the repo does nothing on its own. `daily_mma.py` has
no import of them, which is why your run produced the same ML/DIST board
as always.

## Three lines

```python
from mbm_mma_method import MethodBoard

mb  = MethodBoard(data_dir="data")     # loads + fits once, ~90s
out = mb.board(slate)                  # slate = your ESPN pull
print(mb.render(out))
```

Slate entries:

```python
{"date": "2026-08-22", "a": "Chris Padilla", "b": "Nasrat Haqparast",
 "division": "Lightweight", "rounds": 3, "p_win_a": 0.561}
```

`mb.checks(out)` returns MBM-CHECK tuples in your existing format.
Model tag is `v2-method`, register it in your shipping heads.

## Pass p_win_a. This is the whole ballgame.

The 6-way log loss decomposes exactly:

```
-log P(A by KO) = -log P(A wins) + -log P(KO | A wins)
```

Verified, identity holds to floating point. So the winner marginal flows
through **1:1** and improving it is free.

```
your daily_mma.py moneyline (n=329)   0.6505
elo currently feeding the hybrid      0.6866
6-way with elo                        1.6718
6-way with your p_win_a               1.6357   <- projected
```

That single swap is worth more than everything the hazard model does on
the method side (−0.0124). If `p_win_a` is absent the module falls back
to Elo and the `winner_source` check flags it.

## What this board does NOT replace

Your existing markets are better where they overlap:

| Market | Yours | This model |
|---|---|---|
| moneyline | **0.6505** vs 0.6934 base | 0.6866 (elo) |
| distance | **0.6748** vs 0.6881 base | 0.6812 vs 0.6838 |

Keep both. This board adds the KO-versus-SUB split, which neither of
your current markets touches, and which is where the 3.5–4.2x class lift
sits. Post it alongside ML and DIST, not instead.

## Dry run on your 2026-08-22 slate

Using your p_win_a from the log:

```
MBM-CHECK PASS freshness | ufcstats mirror current to 2026-08-15 (2d)
MBM-CHECK PASS model_identity | tag v2-method
MBM-CHECK FAIL name_resolution | 3 unmatched, 0 fuzzy
MBM-CHECK PASS no_conviction | 1 pick(s), max repeat 1
MBM-CHECK PASS winner_source | ['p_win_a']
MBM-CHECK PASS trust_band | all picks within 0.08-0.26

=== MBM MMA method board ===
2026-08-22  Anthony Hernandez vs Gregory Rodrigues
  Anthony Hernandez by SUB           12.9%  find 7.73 / +673 or better
(5 no edge over base rates, lift < 1.35x)
(4 withheld on thin record (<3 bouts))
(3 unmatched to ufcstats roster)
```

One pick from thirteen bouts. That is the rule working, not failing. At
1.35x lift the historical posting rate is 9–14%.

## The one thing you have to fix first

`name_resolution` fails: Anthony Wint, Marcio Barbosa, Gauge Young and
their opponents are not in the ufcstats roster at all.

Your pipeline pulls the slate from ESPN. This model keys on ufcstats
names. They disagree, and your own log already shows it:

```
Jan Blachowicz          -> jan b achowicz          (encoding)
Diego Ferreira          -> carlos diego ferreira   (given name)
Eric Mcconico           -> eric mcconico jr        (suffix)
Chidi Njokuani vs Geoff Neal   (ESPN)
Chidi Njokuani vs Joel Alvarez (ufcstats)          <- different opponent
```

That last one is not a name problem, it is a card-change problem: one
source has a stale opponent. Ten of your bouts went ungraded for this
reason.

The module handles suffixes, punctuation, fuzzy match at 0.84, and
unique-surname fallback. It cannot invent a fighter who is not in the
mirror, which is the right failure: debutants have no data anyway and
would be withheld regardless.

Build a persistent `name_map.csv` (espn_name → ufcstats_name), have the
module log every fuzzy and unmatched hit, and hand-resolve the tail once.
That fixes your existing grading gap at the same time.

## Order of work

1. `name_map.csv` bridge, ESPN → ufcstats. Fixes both boards.
2. Pass `p_win_a` into `mb.board()`. Worth 0.036 for one argument.
3. Register `v2-method` in your shipping heads.
4. Post as a third block under ML and DIST.
5. Log every pick with the model tag and grade forward. You need ~300
   fights before the result means anything.

## Still open from SHIPPING.md

Predictor Platt is fitted in-sample here, slightly optimistic. Pull the
coefficients from `backtest.py` instead. Per-division reliability is
still unmeasured. Neither blocks posting.
