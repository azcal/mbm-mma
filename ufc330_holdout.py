"""
Clean single-event holdout: UFC 330 (2026-08-15).

The model is trained ONLY on fights strictly before the event date.
Career features, Elo, hazard shape, Platt coefficients and division base
rates are all as-of 2026-08-14. Nothing from the card leaks in.

Twelve fights is an anecdote, not evidence. Read it as a smoke test.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit

import mma_v2 as V
from mma_v2 import estimate_shape, cum, fit_v2, predict_v2, elo_table
from mma_method_model import (
    load, build_fights, build_roundstats, build_dataset, randomize_corners,
    new_state, features, fix_kd15, Scaler, CLASSES, FEAT_KEYS,
)
from block_ridge import design_i, make_pen
from mma_predict import replay, ipf, fair_american, resolve, LAM_MAIN, LAM_INT, TRUST_LO, TRUST_HI

EVENT = "UFC 330"
CUTOFF = pd.Timestamp("2026-08-15")
LIFT_MIN = 1.35

raw = load("/tmp")
all_fights = build_fights(raw)
rstats = build_roundstats(raw)

card = all_fights[all_fights["EVENT"].str.contains(EVENT, na=False)].copy()
train_f = all_fights[all_fights["date"] < CUTOFF].copy()
print(f"training on {len(train_f)} fights through {train_f['date'].max().date()}")
print(f"holdout: {len(card)} bouts on {card['date'].iloc[0].date()}\n")

ds = randomize_corners(build_dataset(train_f, rstats))
divs = sorted(ds["division"].unique())
phik, phis = estimate_shape(train_f, "KO"), estimate_shape(train_f, "SUB")
Ck, Cs = cum(phik), cum(phis)

mats = design_i(ds, divs)
sc = [Scaler().fit(m) for m in mats]
mats_s = tuple(s.t(m) for s, m in zip(sc, mats))
theta = fit_v2(mats_s, ds["end_min"].values, ds["max_min"].values, ds["target"].values,
               make_pen(mats_s, LAM_MAIN, LAM_INT), Ck, Cs, phik, phis)

et = elo_table(train_f)
kk = {(r.EVENT, r.BOUT): (r.f1, r.elo1, r.elo2) for r in et.itertuples(index=False)}
ea = np.array([kk[(r.event, r.bout)][1] if r.A == kk[(r.event, r.bout)][0]
               else kk[(r.event, r.bout)][2] for r in ds.itertuples(index=False)])
eb = np.array([kk[(r.event, r.bout)][2] if r.A == kk[(r.event, r.bout)][0]
               else kk[(r.event, r.bout)][1] for r in ds.itertuples(index=False)])
Pin, _ = predict_v2(theta, mats_s, ds["max_min"].values, Ck, Cs, phik, phis)
Hin = ipf(Pin, expit((ea - eb) / 173.7))
Z = np.zeros_like(Hin)
Z[np.arange(len(ds)), [CLASSES.index(v) for v in ds["target"]]] = 1
platt = []
for j in range(6):
    x = logit(np.clip(Hin[:, j], 1e-6, 1 - 1e-6)); a = Z[:, j]

    def f(ab, x=x, a=a):
        z = expit(ab[0] + ab[1] * x)
        return -(a * np.log(z + 1e-12) + (1 - a) * np.log(1 - z + 1e-12)).mean()
    platt.append(minimize(f, [0.0, 1.0], method="Nelder-Mead").x)

base_rates = {}
for (dv, mx), g in ds.groupby(["division", "max_min"]):
    if len(g) >= 40:
        v = g["method"].value_counts(normalize=True)
        base_rates[(dv, 5 if mx > 20 else 3)] = (v.get("KO", .32), v.get("SUB", .19),
                                                 max(1 - v.get("KO", .32) - v.get("SUB", .19), 1e-6))
gv = ds["method"].value_counts(normalize=True)
base_global = (gv.get("KO", .32), gv.get("SUB", .19),
               max(1 - gv.get("KO", .32) - gv.get("SUB", .19), 1e-6))

state, prior, elo = replay(train_f, rstats)
roster = set(state)

rows = []
for x in card.itertuples(index=False):
    a, b = x.f1, x.f2
    rounds = 5 if x.max_sec == 1500 else 3
    na, _ = resolve(a, roster); nb, _ = resolve(b, roster)
    sa = state.get(na, new_state()); sb = state.get(nb, new_state())
    fa, fb = features(sa, prior), features(sb, prior)
    fa["kd15"], fb["kd15"] = fix_kd15(sa, prior), fix_kd15(sb, prior)
    row = {"division": x.division, "max_min": 25.0 if rounds == 5 else 15.0}
    for kx in FEAT_KEYS:
        row["A_" + kx], row["B_" + kx] = fa[kx], fb[kx]
    df = pd.DataFrame([row])
    m2 = tuple(s.t(m) for s, m in zip(sc, design_i(df, divs)))
    P, S = predict_v2(theta, m2, df["max_min"].values, Ck, Cs, phik, phis)
    pa = expit((elo.get(na, 1500.0) - elo.get(nb, 1500.0)) / 173.7)
    H = ipf(P, np.array([pa]))
    Q = np.array([expit(platt[j][0] + platt[j][1] * logit(np.clip(H[0, j], 1e-6, 1 - 1e-6)))
                  for j in range(6)])
    Q = Q / Q.sum()
    vc = base_rates.get((x.division, rounds), base_global)
    bq = np.array([vc[0] * pa, vc[1] * pa, vc[2] * pa,
                   vc[0] * (1 - pa), vc[1] * (1 - pa), vc[2] * (1 - pa)])
    bq = bq / bq.sum()
    actual = ("A_" if x.winner == a else "B_") + x.method
    rows.append(dict(a=a, b=b, na=na, nb=nb, div=x.division, rounds=rounds, Q=Q, B=bq,
                     actual=actual, ai=CLASSES.index(actual),
                     nfa=int(sa["fights"]), nfb=int(sb["fights"]),
                     res=f"{x.winner} by {x.method} R{x.ROUND} {x.TIME}"))

print("=" * 100)
print("FULL DISTRIBUTION vs REALITY")
print("=" * 100)
print(f"{'bout':<38} {'actual outcome':<30} {'model p':>8} {'base p':>8} {'rank':>5}")
print("-" * 100)
for r in rows:
    rank = int((r["Q"] > r["Q"][r["ai"]]).sum()) + 1
    print(f"{r['a'][:17]+' v '+r['b'][:17]:<38} {r['res'][:29]:<30} "
          f"{r['Q'][r['ai']]:>7.1%} {r['B'][r['ai']]:>8.1%} {rank:>4}/6")

hit1 = sum(1 for r in rows if r["Q"].argmax() == r["ai"])
Qs = np.array([r["Q"] for r in rows]); Bs = np.array([r["B"] for r in rows])
ai = np.array([r["ai"] for r in rows])
mll = -np.log(np.clip(Qs[np.arange(len(rows)), ai], 1e-15, 1)).mean()
bll = -np.log(np.clip(Bs[np.arange(len(rows)), ai], 1e-15, 1)).mean()
print(f"\n  argmax correct        {hit1}/{len(rows)}")
print(f"  6-way log loss model  {mll:.4f}")
print(f"  6-way log loss base   {bll:.4f}   delta {mll - bll:+.4f}")
mm = {"KO": [0, 3], "SUB": [1, 4], "DEC": [2, 5]}
pk = np.array([[r["Q"][v].sum() for v in mm.values()] for r in rows])
bk = np.array([[r["B"][v].sum() for v in mm.values()] for r in rows])
mi = np.array([list(mm).index(r["actual"].split("_")[1]) for r in rows])
print(f"  3-way method  model   {-np.log(pk[np.arange(len(rows)), mi]).mean():.4f}   "
      f"base {-np.log(bk[np.arange(len(rows)), mi]).mean():.4f}")
print(f"  predicted finishes    {(1 - pk[:, 2]).sum():.1f}   actual {(mi != 2).sum()}")

print("\n" + "=" * 100)
print(f"WHAT THE BOARD WOULD HAVE POSTED (lift >= {LIFT_MIN}x, band {TRUST_LO}-{TRUST_HI})")
print("=" * 100)
posted = []
for r in rows:
    lift = r["Q"] / np.maximum(r["B"], 1e-9)
    lift = np.where((r["Q"] >= TRUST_LO) & (r["Q"] <= TRUST_HI), lift, -1.0)
    j = int(lift.argmax())
    thin = min(r["nfa"], r["nfb"]) < 3
    if lift[j] >= LIFT_MIN and not thin:
        posted.append((r, j, lift[j]))
if not posted:
    print("  no qualifying picks")
else:
    print(f"{'bout':<38} {'pick':<32} {'prob':>6} {'fair':>6} {'lift':>6}  result")
    print("-" * 100)
    for r, j, lf in sorted(posted, key=lambda t: -t[2]):
        c = CLASSES[j]
        who = r["a"] if c.startswith("A_") else r["b"]
        won = "HIT " if j == r["ai"] else "miss"
        print(f"{r['a'][:17]+' v '+r['b'][:17]:<38} {who+' by '+c.split('_')[1]:<32} "
              f"{r['Q'][j]:>5.1%} {fair_american(r['Q'][j]):>6} {lf:>5.2f}x  {won} ({r['res'][:26]})")
    hits = sum(1 for r, j, _ in posted if j == r["ai"])
    ep = sum(r["Q"][j] for r, j, _ in posted)
    print(f"\n  posted {len(posted)}/{len(rows)} bouts   hits {hits}   expected {ep:.2f}")
    thin = [r for r in rows if min(r['nfa'], r['nfb']) < 3]
    if thin:
        print(f"  withheld for thin data: {len(thin)}")
        for r in thin:
            print(f"    {r['a']} ({r['nfa']}) vs {r['b']} ({r['nfb']})")
