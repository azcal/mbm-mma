"""
Which selection rule should the board use?

argmax posts whatever outcome is most likely, which is dominated by
whichever outcome has the highest base rate. That is decisions, and a
decision pick carries almost no information: you would have guessed it
without the model.

What we want is the outcome where the model departs most from the
no-model guess, subject to landing in the calibrated trust band.

Rules compared, scored on the stated objective: calibrated (actual >=
predicted) with real lift over base rates.
"""
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.optimize import minimize
from scipy.special import logit

import mma_v2 as V
from mma_v2 import estimate_shape, cum, fit_v2, predict_v2, elo_table, bench
from mma_method_model import (
    load, build_fights, build_roundstats, build_dataset, randomize_corners,
    Scaler, CLASSES,
)
from block_ridge import design_i, make_pen

LO, HI = 0.08, 0.26
rng = np.random.default_rng(0)

raw = load("/tmp")
fights = build_fights(raw)
ds = randomize_corners(build_dataset(fights, build_roundstats(raw)))
et = elo_table(fights)
k = {(r.EVENT, r.BOUT): (r.f1, r.elo1, r.elo2) for r in et.itertuples(index=False)}
ds["elo_A"] = [k[(r.event, r.bout)][1] if r.A == k[(r.event, r.bout)][0]
               else k[(r.event, r.bout)][2] for r in ds.itertuples(index=False)]
ds["elo_B"] = [k[(r.event, r.bout)][2] if r.A == k[(r.event, r.bout)][0]
               else k[(r.event, r.bout)][1] for r in ds.itertuples(index=False)]
divs = sorted(ds["division"].unique())


def onehot(y):
    Z = np.zeros((len(y), 6))
    Z[np.arange(len(y)), [CLASSES.index(v) for v in y]] = 1
    return Z


def ipf(P6, pa, iters=40):
    T = np.stack([P6[:, :3], P6[:, 3:]], axis=1); meth = T.sum(1)
    for _ in range(iters):
        T *= (np.stack([pa, 1 - pa], 1) / np.maximum(T.sum(2), 1e-12))[:, :, None]
        T *= (meth / np.maximum(T.sum(1), 1e-12))[:, None, :]
    o = np.concatenate([T[:, 0, :], T[:, 1, :]], axis=1)
    return o / o.sum(1, keepdims=True)


def platt_fit(p, a):
    x = logit(np.clip(p, 1e-6, 1 - 1e-6))

    def f(ab):
        z = expit(ab[0] + ab[1] * x)
        return -(a * np.log(z + 1e-12) + (1 - a) * np.log(1 - z + 1e-12)).mean()
    return minimize(f, [0.0, 1.0], method="Nelder-Mead").x


P, B, Y, YR = [], [], [], []
for yv in sorted(y for y in ds["date"].dt.year.unique() if y >= 2015):
    tr = ds[ds["date"].dt.year < yv]; te = ds[ds["date"].dt.year == yv]
    if len(tr) < 800 or len(te) < 30:
        continue
    ftr = fights[fights["date"].dt.year < yv]
    pk_, ps_ = estimate_shape(ftr, "KO"), estimate_shape(ftr, "SUB")
    C1, C2 = cum(pk_), cum(ps_)
    Ma, Mb = design_i(tr, divs), design_i(te, divs)
    sc = [Scaler().fit(m) for m in Ma]
    Ma = tuple(s.t(m) for s, m in zip(sc, Ma)); Mb = tuple(s.t(m) for s, m in zip(sc, Mb))
    th = fit_v2(Ma, tr["end_min"].values, tr["max_min"].values, tr["target"].values,
                make_pen(Ma, 0.01, 0.3), C1, C2, pk_, ps_)
    p, _ = predict_v2(th, Mb, te["max_min"].values, C1, C2, pk_, ps_)
    P.append(ipf(p, expit((te["elo_A"].values - te["elo_B"].values) / 173.7)))
    B.append(bench("division_elo", tr, te, None))
    Y.append(te["target"].values); YR.append(np.full(len(te), yv))

P = np.vstack(P); B = np.vstack(B); Y = np.concatenate(Y); YR = np.concatenate(YR)
Z = onehot(Y)

uy = sorted(set(YR)); Qs, idx = [], []
for i, yv in enumerate(uy):
    if i < 2:
        continue
    cm, tm = YR < yv, YR == yv
    co = [platt_fit(P[cm, j], Z[cm, j]) for j in range(6)]
    q = np.column_stack([expit(co[j][0] + co[j][1] * logit(np.clip(P[tm, j], 1e-6, 1 - 1e-6)))
                         for j in range(6)])
    Qs.append(q / q.sum(1, keepdims=True)); idx.append(np.where(tm)[0])
idx = np.concatenate(idx)
Q = np.vstack(Qs); Bc = B[idx]; Zc = Z[idx]
print(f"evaluated on {len(idx)} fights\n")


# ------------------------------------------------------------------
def rule_argmax(Q, Bc):
    return Q.argmax(1), np.ones(len(Q), bool)


def rule_no_dec(Q, Bc):
    M = Q.copy(); M[:, [2, 5]] = -1
    return M.argmax(1), np.ones(len(Q), bool)


def rule_lift_band(Q, Bc):
    R = Q / np.maximum(Bc, 1e-9)
    R = np.where((Q >= LO) & (Q <= HI), R, -1)
    sel = R.argmax(1)
    return sel, R.max(1) > 0


def rule_lift_free(Q, Bc):
    R = Q / np.maximum(Bc, 1e-9)
    return R.argmax(1), np.ones(len(Q), bool)


def rule_absedge_band(Q, Bc):
    D = np.where((Q >= LO) & (Q <= HI), Q - Bc, -9)
    return D.argmax(1), D.max(1) > -9


def rule_lift_band_min(Q, Bc, thr=1.35):
    R = Q / np.maximum(Bc, 1e-9)
    R = np.where((Q >= LO) & (Q <= HI), R, -1)
    return R.argmax(1), R.max(1) >= thr


RULES = [("argmax", rule_argmax), ("argmax no-DEC", rule_no_dec),
         ("max lift, band", rule_lift_band), ("max lift, free", rule_lift_free),
         ("max abs edge, band", rule_absedge_band),
         ("max lift, band, >=1.35x", rule_lift_band_min)]

print(f"{'rule':<24} {'n':>5} {'cov':>6} {'pred':>7} {'actual':>7} {'gap':>7} "
      f"{'base':>7} {'LIFT':>6}  KO/SUB/DEC")
print("-" * 92)
out = {}
for nm, fn in RULES:
    sel, ok = fn(Q, Bc)
    i = np.where(ok)[0]
    p = Q[i, sel[i]]; a = Zc[i, sel[i]]; b = Bc[i, sel[i]]
    comp = [np.isin(sel[i], c).mean() for c in ([0, 3], [1, 4], [2, 5])]
    lift = a.mean() / max(b.mean(), 1e-9)
    boot = np.array([(Zc[i[r], sel[i[r]]].mean() - Q[i[r], sel[i[r]]].mean())
                     for r in (rng.integers(0, len(i), len(i)) for _ in range(3000))])
    out[nm] = dict(n=len(i), pred=p.mean(), act=a.mean(), lift=lift,
                   gap=a.mean() - p.mean(), gap_lo=np.percentile(boot, 2.5))
    print(f"{nm:<24} {len(i):>5} {len(i)/len(Q):>5.0%} {p.mean():>7.3f} {a.mean():>7.3f} "
          f"{a.mean()-p.mean():>+7.3f} {b.mean():>7.3f} {lift:>5.2f}x  "
          f"{comp[0]:.0%}/{comp[1]:.0%}/{comp[2]:.0%}")

print("\ncalibration bar: actual >= predicted (gap >= 0), CI lower bound shown")
for nm in out:
    o = out[nm]
    print(f"  {nm:<24} gap {o['gap']:+.4f}  95% lower {o['gap_lo']:+.4f}  "
          f"{'PASS' if o['gap_lo'] > -0.02 else 'FAIL'}")

np.save("/home/claude/Q_sel.npy", Q); np.save("/home/claude/B_sel.npy", Bc)
np.save("/home/claude/Z_sel.npy", Zc)

print("\nCOVERAGE / LIFT CURVE for 'max lift, band, >= T'")
print(f"  {'T':>6} {'n':>6} {'cov':>6} {'pred':>7} {'actual':>7} {'gap':>7} {'LIFT':>6}  KO/SUB/DEC")
for T in [1.0, 1.10, 1.20, 1.30, 1.35, 1.45, 1.60, 1.80, 2.00]:
    R = Q / np.maximum(Bc, 1e-9)
    R = np.where((Q >= LO) & (Q <= HI), R, -1)
    sel = R.argmax(1); ok = R.max(1) >= T
    i = np.where(ok)[0]
    if len(i) < 40: continue
    pp = Q[i, sel[i]]; aa = Zc[i, sel[i]]; bb = Bc[i, sel[i]]
    cc = [np.isin(sel[i], c).mean() for c in ([0,3],[1,4],[2,5])]
    print(f"  {T:>6.2f} {len(i):>6} {len(i)/len(Q):>5.0%} {pp.mean():>7.3f} {aa.mean():>7.3f} "
          f"{aa.mean()-pp.mean():>+7.3f} {aa.mean()/max(bb.mean(),1e-9):>5.2f}x  "
          f"{cc[0]:.0%}/{cc[1]:.0%}/{cc[2]:.0%}")

best = max(out.items(), key=lambda kv: kv[1]["lift"] if kv[1]["gap"] > -0.01 else -9)
print(f"\nbest by lift subject to calibration: {best[0]}  "
      f"lift {best[1]['lift']:.2f}x  gap {best[1]['gap']:+.4f}  n={best[1]['n']}")
