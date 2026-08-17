# Board selection rule

## Why not argmax

Argmax posts whichever outcome is most likely. That is dominated by
whichever outcome has the highest base rate, which is decisions. A
decision pick is something you would have guessed without the model.

Backtested on 4,695 fights:

| Rule | n | Cov | Pred | Actual | Gap | LIFT | KO/SUB/DEC |
|---|---|---|---|---|---|---|---|
| argmax | 4695 | 100% | 0.281 | 0.277 | −0.004 | **1.03x** | 18/0/82 |
| argmax no-DEC | 4695 | 100% | 0.179 | 0.196 | +0.017 | 1.17x | 86/14/0 |
| max lift, band | 4683 | 100% | 0.163 | 0.179 | +0.016 | 1.30x | 53/23/24 |
| max abs edge, band | 4683 | 100% | 0.169 | 0.184 | +0.015 | 1.29x | 53/20/28 |
| **max lift, band, ≥1.35x** | **639** | **14%** | **0.150** | **0.185** | **+0.035** | **1.86x** | **38/47/15** |

Argmax scores 1.03x lift. It publishes base rates with extra steps, and
it never once selected a submission in 4,695 fights because submission
tops out near 17% and something always outranks it.

## The shipped rule

Post the outcome where the model departs most from the no-model guess:

```
lift = P_model / P_base        P_base = division x rounds method rate x Elo
```

Restrict to the calibrated trust band (0.08–0.26), post only if lift
≥ 1.35x. Otherwise post nothing for that bout.

Result: **1.86x lift, actual 18.5% against 15.0% predicted.** Beats the
"or more" bar by 3.5 points with a bootstrap lower bound of +0.0055.
Submissions become the largest slice at 47%, which is where the 3.5–4.2x
class lift lives.

## Coverage knob

`--lift T` trades volume for signal. Curve on the same 4,695 fights:

| T | n | Cov | Pred | Actual | Gap | LIFT | KO/SUB/DEC |
|---|---|---|---|---|---|---|---|
| 1.00 | 4323 | 92% | 0.167 | 0.181 | +0.014 | 1.31x | 53/22/26 |
| 1.20 | 2080 | 44% | 0.165 | 0.191 | +0.026 | 1.53x | 49/28/22 |
| 1.30 | 943 | 20% | 0.155 | 0.180 | +0.026 | 1.68x | 43/40/17 |
| **1.35** | **639** | **14%** | **0.150** | **0.185** | **+0.035** | **1.86x** | 38/47/15 |
| 1.45 | 340 | 7% | 0.138 | 0.168 | +0.029 | 2.00x | 35/57/8 |
| 1.60 | 181 | 4% | 0.131 | 0.166 | +0.034 | 2.30x | 32/64/4 |
| 1.80 | 75 | 2% | 0.130 | 0.160 | +0.030 | 2.55x | 28/68/4 |

Calibration holds across the whole range, so this is purely a
volume-versus-signal choice. 1.35 is the default because lift climbs
steeply to that point and then flattens while coverage keeps collapsing.
Below 1.20 the board fills with decisions again.

At 14% coverage a 12-fight card yields one or two picks. Some cards will
yield none. That is the rule working.

## Example

```
BOUT                                PICK                         PROB   FAIR   DEC   BASE   LIFT
Gerald Meerschaert vs Brad Tavares  Gerald Meerschaert by SUB  11.5%   +767  8.67  5.8%  2.00x
Sean Strickland vs Kelvin Gastelum  Kelvin Gastelum by DEC     22.9%   +336  4.36 15.4%  1.48x
Derrick Lewis vs Marcin Tybura      Marcin Tybura by DEC       22.3%   +348  4.48 16.0%  1.39x

2 bout(s) no edge over base rates (lift < 1.35x):
  Charles Oliveira vs Dan Hooker   best 1.28x
  Angela Hill vs Loopy Godinez   best 0.75x

1 bout(s) withheld, insufficient UFC history:
  Islam Dulatov vs Vanilto Antunes
```

Prices are no-vig: `100(1-p)/p`. No shading.

Note what changed against the argmax version of the same card. Oliveira
by DEC at 36.0% was the headline pick; it is now dropped, because 36% is
only 1.28x the base rate for a lightweight decision with Oliveira
favoured. The board instead leads with Meerschaert by submission at
11.5%, a 2.00x departure from the 5.8% base rate, which is a genuine
model read. Meerschaert is a submission specialist facing a fighter
whose recorded vulnerability is there.

Low probability, high information. That is the trade the rule makes.
