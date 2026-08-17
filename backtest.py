"""
The backtest. One command, one pass, one versioned artifact.

    python3 backtest.py

Writes backtest_results.json (machine-readable, commit it) and prints a
report. Exit code 1 if any pre-registered bar is breached, so it can gate
a workflow.

Everything the model claims lives here. run_v2.py, calibration.py and
evaluate.py were research scripts that each re-ran the walk separately;
this runs it once and reports all of it.

PRE-REGISTERED BARS -- set before live use, breach means retire:

  B1  6-way log loss beats division x Elo bench
  B2  pooled ECE (calibrated) <= 0.020
  B3  reliability < resolution on every class
  B4  SUB lift (top20/bottom20) >= 2.50x
  B5  KO  lift >= 2.00x
  B6  calibration bands 0.08-0.26: observed >= predicted - 0.010
  B7  shuffled-label control shows no edge (see checks.py)

Rationale for the bars is in CALIBRATION.md. They sit roughly 20% below
observed values so ordinary year-to-year noise does not trip them.
"""
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit

import mma_v2 as V
from mma_v2 import estimate_shape, cum, fit_v2, predict_v2, elo_table, bench, ll_of
from mma_method_model import (
    load, build_fights, build_roundstats, build_dataset, randomize_corners,
    Scaler, CLASSES,
)
from block_ridge import design_i, make_pen

DATA = os.environ.get("MMA_DATA_DIR", "/tmp")
LAM_MAIN, LAM_INT = 0.01, 0.3
START_YEAR = 2015
OUT = "/home/claude/backtest_results.json"

BARS = dict(b1_beats_elo_bench=True, b2_max_ece=0.020, b4_min_sub_lift=2.50,
            b5_min_ko_lift=2.00, b6_max_shortfall=0.010)


def onehot(y):
    Z = np.zeros((len(y), 6))
    Z[np.arange(len(y)), [CLASSES.index(v) for v in y]] = 1
    return Z


def ipf(P6, pa, iters=40):
    T = np.stack([P6[:, :3], P6[:, 3:]], axis=1)
    meth = T.sum(1)
    for _ in range(iters):
        T *= (np.stack([pa, 1 - pa], 1) / np.maximum(T.sum(2), 1e-12))[:, :, None]
        T *= (meth / np.maximum(T.sum(1), 1e-12))[:, None, :]
    out = np.concatenate([T[:, 0, :], T[:, 1, :]], axis=1)
    return out / out.sum(1, keepdims=True)


def murphy(p, a, bins=10):
    base = a.mean()
    e = np.quantile(p, np.linspace(0, 1, bins + 1)); e[0], e[-1] = -1e-9, 1 + 1e-9
    rel = res = 0.0
    for i in range(bins):
        m = (p >= e[i]) & (p < e[i + 1])
        if m.sum():
            w = m.sum() / len(p)
            rel += w * (p[m].mean() - a[m].mean()) ** 2
            res += w * (a[m].mean() - base) ** 2
    return dict(brier=float(((p - a) ** 2).mean()), rel=float(rel),
                res=float(res), unc=float(base * (1 - base)))


def ece(p, a, bins=10):
    e = np.quantile(p, np.linspace(0, 1, bins + 1)); e[0], e[-1] = -1e-9, 1 + 1e-9
    t = 0.0
    for i in range(bins):
        m = (p >= e[i]) & (p < e[i + 1])
        if m.sum():
            t += m.sum() / len(p) * abs(p[m].mean() - a[m].mean())
    return float(t)


def platt_fit(p, a):
    x = logit(np.clip(p, 1e-6, 1 - 1e-6))

    def f(ab):
        z = expit(ab[0] + ab[1] * x)
        return -(a * np.log(z + 1e-12) + (1 - a) * np.log(1 - z + 1e-12)).mean()
    return minimize(f, [0.0, 1.0], method="Nelder-Mead").x


def platt_apply(P, coef):
    Q = np.column_stack([expit(coef[j][0] + coef[j][1] *
                               logit(np.clip(P[:, j], 1e-6, 1 - 1e-6))) for j in range(6)])
    return Q / Q.sum(1, keepdims=True)


# ----------------------------------------------------------------------
def main():
    t0 = datetime.now(timezone.utc)
    raw = load(DATA)
    fights = build_fights(raw)
    ds = randomize_corners(build_dataset(fights, build_roundstats(raw)))
    et = elo_table(fights)
    k = {(r.EVENT, r.BOUT): (r.f1, r.elo1, r.elo2) for r in et.itertuples(index=False)}
    ds["elo_A"] = [k[(r.event, r.bout)][1] if r.A == k[(r.event, r.bout)][0]
                   else k[(r.event, r.bout)][2] for r in ds.itertuples(index=False)]
    ds["elo_B"] = [k[(r.event, r.bout)][2] if r.A == k[(r.event, r.bout)][0]
                   else k[(r.event, r.bout)][1] for r in ds.itertuples(index=False)]
    divs = sorted(ds["division"].unique())

    years = sorted(y for y in ds["date"].dt.year.unique() if y >= START_YEAR)
    P, Y, YR, BE, PER = [], [], [], [], []
    for yr in years:
        tr = ds[ds["date"].dt.year < yr]; te = ds[ds["date"].dt.year == yr]
        if len(tr) < 800 or len(te) < 30:
            continue
        ftr = fights[fights["date"].dt.year < yr]
        pk_, ps_ = estimate_shape(ftr, "KO"), estimate_shape(ftr, "SUB")
        C1, C2 = cum(pk_), cum(ps_)
        Ma, Mb = design_i(tr, divs), design_i(te, divs)
        sc = [Scaler().fit(m) for m in Ma]
        Ma = tuple(s.t(m) for s, m in zip(sc, Ma)); Mb = tuple(s.t(m) for s, m in zip(sc, Mb))
        th = fit_v2(Ma, tr["end_min"].values, tr["max_min"].values, tr["target"].values,
                    make_pen(Ma, LAM_MAIN, LAM_INT), C1, C2, pk_, ps_)
        p, _ = predict_v2(th, Mb, te["max_min"].values, C1, C2, pk_, ps_)
        pa = expit((te["elo_A"].values - te["elo_B"].values) / 173.7)
        h = ipf(p, pa)
        b = bench("division_elo", tr, te, None)
        P.append(h); Y.append(te["target"].values); BE.append(b)
        YR.append(np.full(len(te), yr))
        PER.append(dict(year=int(yr), n=int(len(te)),
                        model_ll=float(ll_of(h, te["target"].values, "6way").mean()),
                        bench_ll=float(ll_of(b, te["target"].values, "6way").mean())))

    P = np.vstack(P); Y = np.concatenate(Y); YR = np.concatenate(YR); BE = np.vstack(BE)
    Z = onehot(Y)

    # walk-forward Platt: calibrator only ever sees earlier years
    uy = sorted(set(YR))
    Qs, idx = [], []
    for i, yv in enumerate(uy):
        if i < 2:
            continue
        cm, tm = YR < yv, YR == yv
        coef = [platt_fit(P[cm, j], Z[cm, j]) for j in range(6)]
        Qs.append(platt_apply(P[tm], coef)); idx.append(np.where(tm)[0])
    idx = np.concatenate(idx); Q = np.vstack(Qs)
    Zc, Yc = Z[idx], Y[idx]
    yi = np.array([CLASSES.index(v) for v in Yc])

    R = {}
    R["meta"] = dict(
        run_utc=t0.isoformat(), n_fights_total=int(len(fights)),
        n_oos=int(len(Y)), n_calibrated=int(len(idx)),
        oos_years=[int(u) for u in uy], latest_event=str(fights["date"].max().date()),
        lam_main=LAM_MAIN, lam_int=LAM_INT,
        data_sha=hashlib.sha256(
            open(f"{DATA}/ufc_fight_results.csv", "rb").read()).hexdigest()[:16],
    )
    R["per_year"] = PER
    R["ladder"] = {n: float(ll_of(bench(n, ds[ds["date"].dt.year < 2015], ds, None)
                                  if False else BE, Y, "6way").mean())
                   for n in ["division_elo"]}
    R["markets"] = {}
    for mk in ["6way", "4way", "3way_method", "distance", "winner"]:
        lm = ll_of(P, Y, mk).mean(); lb = ll_of(BE, Y, mk).mean()
        R["markets"][mk] = dict(model=float(lm), bench=float(lb), delta=float(lm - lb))

    R["calibration"] = {}
    for j, c in enumerate(CLASSES):
        m = murphy(Q[:, j], Zc[:, j])
        lo = Q[:, j] <= np.quantile(Q[:, j], .2); hi = Q[:, j] >= np.quantile(Q[:, j], .8)
        R["calibration"][c] = dict(
            pred=float(Q[:, j].mean()), actual=float(Zc[:, j].mean()),
            ece=ece(Q[:, j], Zc[:, j]), lift=float(Zc[hi].mean() / max(Zc[lo, j].mean(), 1e-9))
            if False else float(Zc[hi, j].mean() / max(Zc[lo, j].mean(), 1e-9)), **m)

    fp, fa = Q.ravel(), Zc.ravel()
    edges = [0, .05, .08, .11, .14, .17, .21, .26, .33, 1.01]
    R["bands"] = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (fp >= a) & (fp < b)
        if m.sum() < 40:
            continue
        n = int(m.sum()); ph = float(fa[m].mean()); pr = float(fp[m].mean())
        se = float(np.sqrt(max(ph * (1 - ph), 1e-9) / n))
        R["bands"].append(dict(lo=a, hi=b, n=n, pred=pr, actual=ph,
                               ci=[ph - 1.96 * se, ph + 1.96 * se], shortfall=pr - ph))

    R["pooled"] = dict(
        model_ll=float(-np.log(np.clip(Q[np.arange(len(yi)), yi], 1e-15, 1)).mean()),
        ece=float(np.mean([R["calibration"][c]["ece"] for c in CLASSES])),
        brier=float(((Q - Zc) ** 2).sum(1).mean()))

    # ---------------- bars ----------------
    bars, fails = {}, []
    bars["B1_beats_elo_bench"] = R["markets"]["6way"]["delta"] < 0
    bars["B2_ece"] = R["pooled"]["ece"] <= BARS["b2_max_ece"]
    bars["B3_res_gt_rel"] = all(R["calibration"][c]["res"] > R["calibration"][c]["rel"]
                                for c in CLASSES)
    sub_lift = min(R["calibration"]["A_SUB"]["lift"], R["calibration"]["B_SUB"]["lift"])
    ko_lift = min(R["calibration"]["A_KO"]["lift"], R["calibration"]["B_KO"]["lift"])
    bars["B4_sub_lift"] = sub_lift >= BARS["b4_min_sub_lift"]
    bars["B5_ko_lift"] = ko_lift >= BARS["b5_min_ko_lift"]
    bars["B6_band_shortfall"] = all(
        bd["shortfall"] <= BARS["b6_max_shortfall"]
        for bd in R["bands"] if bd["lo"] >= 0.08 and bd["hi"] <= 0.26)
    R["bars"] = {k: bool(v) for k, v in bars.items()}
    R["bar_values"] = dict(sub_lift=float(sub_lift), ko_lift=float(ko_lift),
                           ece=R["pooled"]["ece"], delta_6way=R["markets"]["6way"]["delta"])
    fails = [k for k, v in bars.items() if not v]

    json.dump(R, open(OUT, "w"), indent=2)

    # ---------------- report ----------------
    print("=" * 74)
    print(f"MMA METHOD MODEL BACKTEST   run {t0:%Y-%m-%d %H:%M}Z   data {R['meta']['data_sha']}")
    print("=" * 74)
    print(f"  fights parsed {R['meta']['n_fights_total']}   out-of-sample {R['meta']['n_oos']}"
          f"   calibrated {R['meta']['n_calibrated']}   latest {R['meta']['latest_event']}")

    print("\n  WALK-FORWARD BY YEAR (6-way log loss)")
    print(f"  {'year':>6} {'n':>5} {'model':>8} {'bench':>8} {'delta':>8}")
    for r in PER:
        print(f"  {r['year']:>6} {r['n']:>5} {r['model_ll']:>8.4f} {r['bench_ll']:>8.4f} "
              f"{r['model_ll'] - r['bench_ll']:>+8.4f}")
    won = sum(1 for r in PER if r["model_ll"] < r["bench_ll"])
    print(f"  beat bench in {won}/{len(PER)} years")

    print("\n  MARKETS (vs division x Elo)")
    for mk, v in R["markets"].items():
        print(f"  {mk:<14} model {v['model']:.4f}  bench {v['bench']:.4f}  "
              f"delta {v['delta']:+.4f}")

    print("\n  CALIBRATION (walk-forward Platt)")
    print(f"  {'class':<7} {'pred':>7} {'actual':>7} {'ECE':>7} {'REL':>9} {'RES':>9} {'lift':>7}")
    for c in CLASSES:
        v = R["calibration"][c]
        print(f"  {c:<7} {v['pred']:>7.4f} {v['actual']:>7.4f} {v['ece']:>7.4f} "
              f"{v['rel']:>9.5f} {v['res']:>9.5f} {v['lift']:>6.2f}x")

    print("\n  RELIABILITY BANDS")
    print(f"  {'band':<12} {'n':>6} {'pred':>7} {'actual':>7} {'shortfall':>10}")
    for bd in R["bands"]:
        flag = "  <-- " if bd["shortfall"] > BARS["b6_max_shortfall"] else ""
        print(f"  {bd['lo']:.2f}-{bd['hi']:.2f}   {bd['n']:>6} {bd['pred']:>7.3f} "
              f"{bd['actual']:>7.3f} {bd['shortfall']:>+10.4f}{flag}")

    print("\n  PRE-REGISTERED BARS")
    for kk, v in R["bars"].items():
        print(f"  [{'PASS' if v else 'FAIL'}] {kk}")
    print(f"\n  sub_lift {sub_lift:.2f}x   ko_lift {ko_lift:.2f}x   "
          f"ece {R['pooled']['ece']:.4f}   6way delta {R['markets']['6way']['delta']:+.4f}")
    print(f"\n  wrote {OUT}")
    print("=" * 74)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
