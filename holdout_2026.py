"""
Rolling event-by-event holdout across 2026.

For every event, the model is retrained on fights strictly before that
event's date, then predicts it. This is stricter than the annual
walk-forward: no event ever sees a same-year fight in training.

Platt coefficients are fitted once on 2015-2025 out-of-sample
predictions and frozen before 2026 begins.

NOTE ON SCOPE: UFC Stats covers UFC only. There is no Bellator, PFL, ONE
or RIZIN in this data, so "top tier MMA" here means the UFC, including
Fight Night and Contender Series cards.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit

import mma_v2 as V
from mma_v2 import estimate_shape, cum, fit_v2, predict_v2, elo_table
from mma_method_model import (
    load, build_fights, build_roundstats, build_dataset, randomize_corners,
    Scaler, CLASSES,
)
from block_ridge import design_i, make_pen
from mma_predict import ipf, fair_american

LAM_MAIN, LAM_INT = 0.01, 0.3
LO, HI, LIFT_MIN = 0.08, 0.26, 1.35
rng = np.random.default_rng(0)

raw = load("/tmp")
fights = build_fights(raw)
ds = randomize_corners(build_dataset(fights, build_roundstats(raw))).reset_index(drop=True)
divs = sorted(ds["division"].unique())

et = elo_table(fights)
kk = {(r.EVENT, r.BOUT): (r.f1, r.elo1, r.elo2) for r in et.itertuples(index=False)}
ds["elo_A"] = [kk[(r.event, r.bout)][1] if r.A == kk[(r.event, r.bout)][0]
               else kk[(r.event, r.bout)][2] for r in ds.itertuples(index=False)]
ds["elo_B"] = [kk[(r.event, r.bout)][2] if r.A == kk[(r.event, r.bout)][0]
               else kk[(r.event, r.bout)][1] for r in ds.itertuples(index=False)]
ds["yr"] = ds["date"].dt.year
ds["pa_elo"] = expit((ds["elo_A"] - ds["elo_B"]) / 173.7)

# design matrices built once on all rows, then sliced. Scaler refit per fold.
ALL = design_i(ds, divs)
Zall = np.zeros((len(ds), 6))
Zall[np.arange(len(ds)), [CLASSES.index(v) for v in ds["target"]]] = 1


def fit_predict(tr_idx, te_idx, train_fights):
    pk_, ps_ = estimate_shape(train_fights, "KO"), estimate_shape(train_fights, "SUB")
    C1, C2 = cum(pk_), cum(ps_)
    Ma = tuple(m[tr_idx] for m in ALL)
    sc = [Scaler().fit(m) for m in Ma]
    Ma = tuple(s.t(m) for s, m in zip(sc, Ma))
    Mb = tuple(s.t(m[te_idx]) for s, m in zip(sc, ALL))
    tr = ds.iloc[tr_idx]; te = ds.iloc[te_idx]
    th = fit_v2(Ma, tr["end_min"].values, tr["max_min"].values, tr["target"].values,
                make_pen(Ma, LAM_MAIN, LAM_INT), C1, C2, pk_, ps_)
    P, _ = predict_v2(th, Mb, te["max_min"].values, C1, C2, pk_, ps_)
    return ipf(P, te["pa_elo"].values)


def base_of(tr_idx, te_idx):
    tr = ds.iloc[tr_idx]; te = ds.iloc[te_idx]
    gv = tr["method"].value_counts(normalize=True)
    g = (gv.get("KO", .32), gv.get("SUB", .19),
         max(1 - gv.get("KO", .32) - gv.get("SUB", .19), 1e-6))
    br = {}
    for (dv, mx), grp in tr.groupby(["division", "max_min"]):
        if len(grp) >= 40:
            v = grp["method"].value_counts(normalize=True)
            br[(dv, mx)] = (v.get("KO", .32), v.get("SUB", .19),
                            max(1 - v.get("KO", .32) - v.get("SUB", .19), 1e-6))
    out = np.zeros((len(te), 6))
    for i, r in enumerate(te.itertuples(index=False)):
        v = br.get((r.division, r.max_min), g)
        pa = r.pa_elo
        out[i] = [v[0] * pa, v[1] * pa, v[2] * pa,
                  v[0] * (1 - pa), v[1] * (1 - pa), v[2] * (1 - pa)]
    return out / out.sum(1, keepdims=True)


# ---- Platt fitted on 2015-2025 OOS, frozen before 2026 ----
print("fitting calibrator on 2015-2025 out-of-sample predictions ...")
Pp, Zp = [], []
for y in range(2015, 2026):
    tri = np.where(ds["yr"].values < y)[0]; tei = np.where(ds["yr"].values == y)[0]
    if len(tri) < 800 or len(tei) < 30:
        continue
    Pp.append(fit_predict(tri, tei, fights[fights["date"].dt.year < y]))
    Zp.append(Zall[tei])
Pp, Zp = np.vstack(Pp), np.vstack(Zp)
PLATT = []
for j in range(6):
    x = logit(np.clip(Pp[:, j], 1e-6, 1 - 1e-6)); a = Zp[:, j]

    def f(ab, x=x, a=a):
        z = expit(ab[0] + ab[1] * x)
        return -(a * np.log(z + 1e-12) + (1 - a) * np.log(1 - z + 1e-12)).mean()
    PLATT.append(minimize(f, [0.0, 1.0], method="Nelder-Mead").x)
print(f"  frozen on {len(Pp)} fights\n")


def cal(P):
    Q = np.column_stack([expit(PLATT[j][0] + PLATT[j][1] *
                               logit(np.clip(P[:, j], 1e-6, 1 - 1e-6))) for j in range(6)])
    return Q / Q.sum(1, keepdims=True)


# ---- rolling event-by-event through 2026 ----
ev = (ds[ds["yr"] == 2026].groupby("event")["date"].first()
      .sort_values(ascending=False))
print(f"{len(ev)} UFC events in 2026, working backwards from {ev.index[0][:28]}\n")

rows, per_event = [], []
for name, d in ev.items():
    tei = np.where((ds["event"] == name).values)[0]
    tri = np.where((ds["date"] < d).values)[0]
    Q = cal(fit_predict(tri, tei, fights[fights["date"] < d]))
    B = base_of(tri, tei)
    te = ds.iloc[tei]
    ai = np.array([CLASSES.index(v) for v in te["target"]])
    mll = -np.log(np.clip(Q[np.arange(len(tei)), ai], 1e-15, 1))
    bll = -np.log(np.clip(B[np.arange(len(tei)), ai], 1e-15, 1))
    per_event.append(dict(event=name, date=d, n=len(tei), m=mll.mean(), b=bll.mean()))
    for i in range(len(tei)):
        rows.append(dict(event=name, date=d, Q=Q[i], B=B[i], ai=ai[i],
                         A=te["A"].iloc[i], Bf=te["B"].iloc[i],
                         nA=te["n_prior_A"].iloc[i], nB=te["n_prior_B"].iloc[i],
                         target=te["target"].iloc[i]))

Qs = np.array([r["Q"] for r in rows]); Bs = np.array([r["B"] for r in rows])
ai = np.array([r["ai"] for r in rows]); n = len(rows)
mll = -np.log(np.clip(Qs[np.arange(n), ai], 1e-15, 1))
bll = -np.log(np.clip(Bs[np.arange(n), ai], 1e-15, 1))

print("=" * 84)
print("PER-EVENT (newest first), 6-way log loss")
print("=" * 84)
print(f"  {'date':<11} {'event':<40} {'n':>3} {'model':>7} {'base':>7} {'delta':>8}")
for e in per_event:
    mark = "" if e["m"] < e["b"] else "  <"
    print(f"  {str(e['date'].date()):<11} {e['event'][:39]:<40} {e['n']:>3} "
          f"{e['m']:>7.4f} {e['b']:>7.4f} {e['m'] - e['b']:>+8.4f}{mark}")
pd.DataFrame(per_event).assign(delta=lambda x: x.m-x.b).to_csv(
    "/home/claude/holdout_2026_events.csv", index=False)
won = sum(1 for e in per_event if e["m"] < e["b"])
print(f"\n  model beat base rates on {won}/{len(per_event)} events")

print("\n" + "=" * 84)
print("2026 AGGREGATE")
print("=" * 84)
d = mll - bll
boot = np.array([d[rng.integers(0, n, n)].mean() for _ in range(8000)])
print(f"  fights                {n}")
print(f"  6-way model log loss  {mll.mean():.4f}")
print(f"  6-way base log loss   {bll.mean():.4f}")
print(f"  delta                 {d.mean():+.4f}   95% CI [{np.percentile(boot,2.5):+.4f}, "
      f"{np.percentile(boot,97.5):+.4f}]   P(>=0) {(boot>=0).mean():.3f}")
mm = {"KO": [0, 3], "SUB": [1, 4], "DEC": [2, 5]}
pk = np.column_stack([Qs[:, v].sum(1) for v in mm.values()])
bk = np.column_stack([Bs[:, v].sum(1) for v in mm.values()])
mi = np.array([list(mm).index(r["target"].split("_")[1]) for r in rows])
print(f"  3-way method model    {-np.log(pk[np.arange(n), mi]).mean():.4f}   "
      f"base {-np.log(bk[np.arange(n), mi]).mean():.4f}")
print(f"  predicted finishes    {(1 - pk[:, 2]).sum():.0f}   actual {(mi != 2).sum()}")

print("\n" + "=" * 84)
print(f"BOARD PICKS (lift >= {LIFT_MIN}x, band {LO}-{HI}, thin data withheld)")
print("=" * 84)
posted = []
for r in rows:
    lift = np.where((r["Q"] >= LO) & (r["Q"] <= HI), r["Q"] / np.maximum(r["B"], 1e-9), -1)
    j = int(lift.argmax())
    if lift[j] >= LIFT_MIN and min(r["nA"], r["nB"]) >= 3:
        posted.append((r, j, lift[j]))

if posted:
    P_ = np.array([r["Q"][j] for r, j, _ in posted])
    H_ = np.array([1.0 if j == r["ai"] else 0.0 for r, j, _ in posted])
    Bb = np.array([r["B"][j] for r, j, _ in posted])
    meth = [CLASSES[j].split("_")[1] for _, j, _ in posted]
    print(f"  posted           {len(posted)} of {n} bouts ({len(posted)/n:.0%})")
    print(f"  predicted hits   {P_.sum():.1f}")
    print(f"  actual hits      {int(H_.sum())}")
    print(f"  mean predicted   {P_.mean():.3f}")
    print(f"  actual rate      {H_.mean():.3f}   gap {H_.mean()-P_.mean():+.3f}")
    print(f"  base rate        {Bb.mean():.3f}   LIFT {H_.mean()/max(Bb.mean(),1e-9):.2f}x")
    k = int(H_.sum()); N = len(H_)
    sims = rng.binomial(N, P_.mean(), 20000)
    print(f"  P(<= {k} hits | model correct) = {(sims<=k).mean():.3f}")
    print(f"\n  {'method':<6} {'n':>4} {'pred':>7} {'hits':>5} {'rate':>7}")
    for mth in ["KO", "SUB", "DEC"]:
        m = np.array([x == mth for x in meth])
        if m.sum():
            print(f"  {mth:<6} {m.sum():>4} {P_[m].mean():>7.3f} {int(H_[m].sum()):>5} {H_[m].mean():>7.3f}")
    print(f"\n  {'date':<11} {'pick':<44} {'prob':>6} {'fair':>6} {'lift':>6} {'':>4}")
    for r, j, lf in sorted(posted, key=lambda t: t[0]["date"], reverse=True):
        c = CLASSES[j]; who = r["A"] if c.startswith("A_") else r["Bf"]
        print(f"  {str(r['date'].date()):<11} {who+' by '+c.split('_')[1]:<44} "
              f"{r['Q'][j]:>5.1%} {fair_american(r['Q'][j]):>6} {lf:>5.2f}x "
              f"{'HIT' if j == r['ai'] else '   ':>4}")
