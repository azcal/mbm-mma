# MBM MMA daily engine (rebuild v1)

Standalone predictive model, no market inputs. Three markets from two heads: moneyline,
goes-the-distance, and winner by method. Frozen through 2025; 2026 was opened once and
the rest of 2026 onward is the forward test.

## Frozen 2026 holdout (UFC, n = 295)

| market | model | floor | base | edge | t | skill |
|---|---|---|---|---|---|---|
| moneyline | 0.6480 | 0.6712 Elo | 0.6932 | +0.0451 | 3.24 | 6.5% |
| moneyline layer over Elo | | | | +0.0231 | 2.15 | |
| goes the distance | 0.6741 | | 0.6924 | +0.0183 | 1.69 | 2.6% |
| winner by method (3-way) | 0.9836 | | 1.0343 | +0.0507 | | 4.9% |

Development window 2018-2025 (n = 3,902) read 4.10% on the moneyline and 2.24% on
distance, so the holdout came in ahead of it on a small sample. Treat the development
figure as the more reliable one.

## Design

**Data.** Wikipedia `{{MMAevent bout}}` templates across UFC, Bellator, PFL and ONE:
10,216 decided bouts with method, round and finish time. Fighter dates of birth come from
Wikipedia infoboxes. ESPN supplies upcoming cards, since Wikipedia only lists fights after
they happen. ufcstats.com, the source every public MMA model uses, now sits behind a
JavaScript proof-of-work challenge and is unusable from a scheduled runner.

**Non-UFC bouts update fighter states but are never evaluated.** Cross-promotion fights are
rare, so a shared rating scale across four organisations would rest on weakly connected
pools with a biased bridge. A fighter's Bellator record is still real information about
that fighter. Same pattern as FCS opponents in the NCAAF board.

**Orientation is alphabetical**, so the base rate is 0.500 and nothing is inherited from a
listing convention. ESPN's red-corner ordering was measured at 0.5353, worth 0.0025 log
loss, and rejected as not worth a second data dependency.

**Method is the parent market.** One head gives P(A wins), a second gives P(method | winner)
as a three-way, and distance derives as `1 - [P(A)*P(finish|A) + P(B)*P(finish|B)]`. The
distance number therefore cannot contradict the method numbers.

**Age is the only fighter attribute that earned inclusion**, worth +0.0082 log loss at
t = 3.43 on the development window. Reach, height and stance measured at zero and were
dropped: reach parsed on 56% of fighters, stance on 37%, and within a weight class the
spread is narrow.

## Why this board is thinner than the team sports

The Elo floor is worth +0.0083 here against +0.1127 in NCAAF. Across 10,216 bouts there are
4,534 distinct fighters, so the average fighter appears about four and a half times in the
entire dataset, and any fight can end with one punch. That is the thinnest state estimation
in the stack, and it is a property of the sport rather than the model.

## Operating

Runs Friday evening after weigh-ins with a Saturday midday refresh that reports only what
moved. Tripwire: last-200 moneyline log-loss edge against base rate, pinged to Telegram.
Draws, no-contests and DQs are 2.9% of bouts and grade as pushes.

Rebuild from scratch: `python mma_rebuild_v1.py` (tunes on the development window, rewrites
`spec.json`, and reopens the holdout, so only run it deliberately).

## Do not trust a backtest that can see who has a birth date on file (2026-11-26)

`data/attrs.csv` was pulled once, in July 2026, from the roster as it stood then. A fighter
still on the roster in 2026 is a fighter who kept winning, so "this fighter has a date of
birth on file" is information from the future for any fight before the pull. Where only one
of the two had an age on file, that fighter won 70.1% of the time in 2024 and 62.7% in 2025.
In 2026, the first year the file could not have known about, it was 52.0%.

The live winner head is a logistic on linear age terms, with a missing age filled at 30, so
it cannot isolate that flag and its 2026 numbers are clean (0.6624 with the age terms,
0.6633 without). Anything that can split on "age is exactly 30" can. A gradient-boosted
version of this head scored 0.023 and 0.014 better than the live model on 2024 and 2025 and
0.009 worse on 2026, and all of the gain was this flag. It was not adopted.

If attributes are ever refreshed or extended, either stamp each row with the date it was
collected and use it only for fights after that date, or judge the change on fights after
the pull only. The "+0.0082 for age" quoted above was measured on years the file could see;
on 2026 the age terms are worth about +0.001.
