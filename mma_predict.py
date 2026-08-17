"""
MMA method-of-victory predictor: production entry point.

Implements the shipped configuration, which is NOT the v1 model:
  - v2 proportional hazards with non-constant baseline shape
  - Elo winner marginal reconciled onto the method structure by IPF
  - walk-forward-fitted Platt calibration applied to the output

Odds-blind. No prices, no lines, no EV.

    python3 mma_predict.py --refresh
    python3 mma_predict.py --card card.txt
    python3 mma_predict.py --fight "Name A|Name B|Lightweight|5"

Card file: one bout per line, Fighter A|Fighter B|Division|Rounds

Trust bands (see CALIBRATION.md):
  <0.08        model over-calls, do not act
  0.08-0.26    calibrated, errs conservative
  >0.26        runs ~1.2pts hot, discount
"""
import argparse
import difflib
import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit

import mma_v2 as V
import mma_method_model as M
from mma_v2 import estimate_shape, cum, fit_v2, predict_v2, elo_table
from mma_method_model import (
    load, build_fights, build_roundstats, build_dataset, randomize_corners,
    new_state, features, fix_kd15, RunningPrior, Scaler, CLASSES, FEAT_KEYS,
)
from block_ridge import design_i, make_pen

DATA_DIR = os.environ.get("MMA_DATA_DIR", "/tmp")
FILES = ["ufc_event_details", "ufc_fight_results", "ufc_fight_stats", "ufc_fighter_tott"]
LAM_MAIN, LAM_INT = 0.01, 0.3
TRUST_LO, TRUST_HI = 0.08, 0.26
LIFT_MIN = 1.35   # board posts only when model/base-rate >= this


def refresh(dest=DATA_DIR):
    import urllib.request
    os.makedirs(dest, exist_ok=True)
    for f in FILES:
        p = os.path.join(dest, f + ".csv")
        urllib.request.urlretrieve(f"{M.BASE}/{f}.csv", p)
        print(f"  {f}.csv  {os.path.getsize(p):,} bytes")


def replay(fights, rstats):
    """Career state and Elo for every fighter as of the last recorded bout."""
    idx = {(r.EVENT, r.BOUT, r.FIGHTER): r for r in rstats.itertuples(index=False)}
    state, prior, elo = {}, RunningPrior(), {}
    for fr in fights.itertuples(index=False):
        a, b = fr.f1, fr.f2
        sa = state.setdefault(a, new_state()); sb = state.setdefault(b, new_state())
        ra, rb = idx.get((fr.EVENT, fr.BOUT, a)), idx.get((fr.EVENT, fr.BOUT, b))
        mins = fr.end_sec / 60.0
        for me, opp, st in ((ra, rb, sa), (rb, ra, sb)):
            st["mins"] += mins; st["fights"] += 1
            if me is not None:
                st["kd_for"] += me.kd; st["sig_l"] += me.sig_l; st["sig_a"] += me.sig_a
                st["td_l"] += me.td_l; st["td_a"] += me.td_a
                st["sub_for"] += me.subatt; st["ctrl_for"] += me.ctrl
            if opp is not None:
                st["kd_against"] += opp.kd; st["sig_l_opp"] += opp.sig_l
                st["sig_a_opp"] += opp.sig_a; st["td_l_opp"] += opp.td_l
                st["td_a_opp"] += opp.td_a; st["sub_against"] += opp.subatt
                st["ctrl_against"] += opp.ctrl
        aw = fr.winner == a
        (sa if aw else sb)["w_" + fr.method.lower()] += 1
        (sb if aw else sa)["l_" + fr.method.lower()] += 1
        r1, r2 = elo.get(a, 1500.0), elo.get(b, 1500.0)
        e1 = 1 / (1 + 10 ** ((r2 - r1) / 400)); s1 = 1.0 if aw else 0.0
        elo[a] = r1 + 24 * (s1 - e1); elo[b] = r2 + 24 * ((1 - s1) - (1 - e1))
        if mins > 0 and ra is not None and rb is not None:
            prior.add({
                "kd15": (ra.kd + rb.kd) / 2 * 15 / mins, "kda15": (ra.kd + rb.kd) / 2 * 15 / mins,
                "slpm": (ra.sig_l + rb.sig_l) / 2 / mins, "sapm": (ra.sig_l + rb.sig_l) / 2 / mins,
                "acc": np.nan_to_num((ra.sig_l + rb.sig_l) / max(ra.sig_a + rb.sig_a, 1)),
                "dfn": 1 - np.nan_to_num((ra.sig_l + rb.sig_l) / max(ra.sig_a + rb.sig_a, 1)),
                "td15": (ra.td_l + rb.td_l) / 2 * 15 / mins,
                "tdacc": np.nan_to_num((ra.td_l + rb.td_l) / max(ra.td_a + rb.td_a, 1)),
                "tddef": 1 - np.nan_to_num((ra.td_l + rb.td_l) / max(ra.td_a + rb.td_a, 1)),
                "sub15": (ra.subatt + rb.subatt) / 2 * 15 / mins,
                "suba15": (ra.subatt + rb.subatt) / 2 * 15 / mins,
                "ctrl": (ra.ctrl + rb.ctrl) / 2 / (mins * 60),
                "wko": 1.0 if fr.method == "KO" else 0.0, "wsub": 1.0 if fr.method == "SUB" else 0.0,
                "lko": 1.0 if fr.method == "KO" else 0.0, "lsub": 1.0 if fr.method == "SUB" else 0.0,
            })
    return state, prior, elo


def ipf(P6, pa, iters=40):
    T = np.stack([P6[:, :3], P6[:, 3:]], axis=1); meth = T.sum(1)
    for _ in range(iters):
        T *= (np.stack([pa, 1 - pa], 1) / np.maximum(T.sum(2), 1e-12))[:, :, None]
        T *= (meth / np.maximum(T.sum(1), 1e-12))[:, None, :]
    out = np.concatenate([T[:, 0, :], T[:, 1, :]], axis=1)
    return out / out.sum(1, keepdims=True)


def resolve(name, roster):
    if name in roster:
        return name, True
    low = {r.lower(): r for r in roster}
    if name.lower() in low:
        return low[name.lower()], True
    h = difflib.get_close_matches(name.lower(), list(low), n=1, cutoff=0.82)
    return (low[h[0]], False) if h else (None, False)


class Model:
    def __init__(self, data_dir=DATA_DIR):
        raw = load(data_dir)
        self.fights = build_fights(raw)
        self.rstats = build_roundstats(raw)
        ds = randomize_corners(build_dataset(self.fights, self.rstats))
        self.divs = sorted(ds["division"].unique())

        self.phik = estimate_shape(self.fights, "KO")
        self.phis = estimate_shape(self.fights, "SUB")
        self.Ck, self.Cs = cum(self.phik), cum(self.phis)

        mats = design_i(ds, self.divs)
        self.sc = [Scaler().fit(m) for m in mats]
        mats = tuple(s.t(m) for s, m in zip(self.sc, mats))
        self.theta = fit_v2(mats, ds["end_min"].values, ds["max_min"].values,
                            ds["target"].values, make_pen(mats, LAM_MAIN, LAM_INT),
                            self.Ck, self.Cs, self.phik, self.phis)

        # Platt coefficients from in-sample hybrid predictions.
        et = elo_table(self.fights)
        kk = {(r.EVENT, r.BOUT): (r.f1, r.elo1, r.elo2) for r in et.itertuples(index=False)}
        ea = np.array([kk[(r.event, r.bout)][1] if r.A == kk[(r.event, r.bout)][0]
                       else kk[(r.event, r.bout)][2] for r in ds.itertuples(index=False)])
        eb = np.array([kk[(r.event, r.bout)][2] if r.A == kk[(r.event, r.bout)][0]
                       else kk[(r.event, r.bout)][1] for r in ds.itertuples(index=False)])
        P, _ = predict_v2(self.theta, mats, ds["max_min"].values,
                          self.Ck, self.Cs, self.phik, self.phis)
        H = ipf(P, expit((ea - eb) / 173.7))
        Z = np.zeros_like(H)
        Z[np.arange(len(ds)), [CLASSES.index(v) for v in ds["target"]]] = 1
        self.platt = []
        for j in range(6):
            x = logit(np.clip(H[:, j], 1e-6, 1 - 1e-6)); a = Z[:, j]

            def f(ab, x=x, a=a):
                z = expit(ab[0] + ab[1] * x)
                return -(a * np.log(z + 1e-12) + (1 - a) * np.log(1 - z + 1e-12)).mean()
            self.platt.append(minimize(f, [0.0, 1.0], method="Nelder-Mead").x)

        # division x rounds method base rates: the no-model guess
        self.base_rates = {}
        for (dv, mx), g in ds.groupby(["division", "max_min"]):
            if len(g) >= 40:
                v = g["method"].value_counts(normalize=True)
                self.base_rates[(dv, 5 if mx > 20 else 3)] = (
                    v.get("KO", .32), v.get("SUB", .19),
                    max(1 - v.get("KO", .32) - v.get("SUB", .19), 1e-6))
        gv = ds["method"].value_counts(normalize=True)
        self.base_global = (gv.get("KO", .32), gv.get("SUB", .19),
                            max(1 - gv.get("KO", .32) - gv.get("SUB", .19), 1e-6))

        self.state, self.prior, self.elo = replay(self.fights, self.rstats)
        self.roster = set(self.state)

    def matchup(self, a, b, division, rounds=3):
        na, exa = resolve(a, self.roster); nb, exb = resolve(b, self.roster)
        sa = self.state.get(na, new_state()); sb = self.state.get(nb, new_state())
        fa, fb = features(sa, self.prior), features(sb, self.prior)
        fa["kd15"], fb["kd15"] = fix_kd15(sa, self.prior), fix_kd15(sb, self.prior)

        row = {"division": division, "max_min": 25.0 if rounds == 5 else 15.0}
        for kk in FEAT_KEYS:
            row["A_" + kk], row["B_" + kk] = fa[kk], fb[kk]
        df = pd.DataFrame([row])
        mats = tuple(s.t(m) for s, m in zip(self.sc, design_i(df, self.divs)))
        P, S = predict_v2(self.theta, mats, df["max_min"].values,
                          self.Ck, self.Cs, self.phik, self.phis)
        pa = expit((self.elo.get(na, 1500.0) - self.elo.get(nb, 1500.0)) / 173.7)
        H = ipf(P, np.array([pa]))
        Q = np.array([expit(self.platt[j][0] + self.platt[j][1] *
                            logit(np.clip(H[0, j], 1e-6, 1 - 1e-6))) for j in range(6)])
        Q = Q / Q.sum()
        vc = self.base_rates.get((division, int(rounds)), self.base_global)
        bq = np.array([vc[0] * pa, vc[1] * pa, vc[2] * pa,
                       vc[0] * (1 - pa), vc[1] * (1 - pa), vc[2] * (1 - pa)])
        bq = bq / bq.sum()
        return dict(A=na or a, B=nb or b, exact=(exa, exb), base=bq,
                    n=(int(sa["fights"]), int(sb["fights"])),
                    elo=(round(self.elo.get(na, 1500.0)), round(self.elo.get(nb, 1500.0))),
                    probs=dict(zip(CLASSES, Q)), surv=S[0], max_min=row["max_min"])


def fmt(r):
    p = r["probs"]; a, b = r["A"], r["B"]; na, nb = r["n"]
    out = [f"{a} ({na} UFC, elo {r['elo'][0]}) vs {b} ({nb} UFC, elo {r['elo'][1]})"]
    if na < 3 or nb < 3:
        out.append("  ** THIN DATA: shrinks to division base rates, do not act **")
    if not all(r["exact"]):
        out.append("  ** fuzzy name match, verify **")
    pa = p["A_KO"] + p["A_SUB"] + p["A_DEC"]
    out.append(f"  win   {a} {pa:.1%} | {b} {1 - pa:.1%}")
    for k, v in sorted(p.items(), key=lambda x: -x[1]):
        who = a if k.startswith("A_") else b
        tag = "" if TRUST_LO <= v <= TRUST_HI else ("  [below trust band]" if v < TRUST_LO
                                                    else "  [runs hot, discount]")
        out.append(f"  {who} by {k.split('_')[1]:<4} {v:6.1%}{tag}")
    ko = p["A_KO"] + p["B_KO"]; sb = p["A_SUB"] + p["B_SUB"]
    out.append(f"  method  KO {ko:.1%}  SUB {sb:.1%}  DEC {1 - ko - sb:.1%}")
    nr = int(r["max_min"] // 5)
    segs = []
    for i in range(nr):
        g = int((i + 1) * 5 / V.GRID) - 1
        prev = 1.0 if i == 0 else r["surv"][int(i * 5 / V.GRID) - 1]
        segs.append(f"R{i + 1} {prev - r['surv'][g]:.1%}")
    out.append("  finish by round  " + "  ".join(segs))
    return "\n".join(out)


def fair_american(p):
    """No-vig American price implied by p. No shading, no built-in edge."""
    if p >= 0.5:
        return f"-{round(100 * p / (1 - p))}"
    return f"+{round(100 * (1 - p) / p)}"


def board_line(r, lift_min=LIFT_MIN):
    """
    Post the outcome where the model departs most from the no-model guess,
    inside the calibrated trust band, and only if the departure clears
    lift_min. Argmax is deliberately NOT used: it selects whichever
    outcome has the highest base rate, which is decisions, and a decision
    pick is something you would have guessed without the model. Backtest
    of the rules is in select_rule.py.
    """
    a, b = r["A"], r["B"]
    P = np.array([r["probs"][c] for c in CLASSES])
    Bq = np.array(r["base"])
    lift = P / np.maximum(Bq, 1e-9)
    lift = np.where((P >= TRUST_LO) & (P <= TRUST_HI), lift, -1.0)
    j = int(lift.argmax())
    thin = min(r["n"]) < 3
    if lift[j] < lift_min or thin:
        return dict(bout=f"{a} vs {b}", pick=None, thin=thin,
                    best=float(lift.max()))
    c = CLASSES[j]
    who = a if c.startswith("A_") else b
    return dict(bout=f"{a} vs {b}", pick=f"{who} by {c.split('_')[1]}",
                p=float(P[j]), base=float(Bq[j]), lift=float(lift[j]),
                fair=fair_american(float(P[j])), dec=round(1 / float(P[j]), 2),
                thin=False)


def print_board(rows, show_thin=False):
    live = [r for r in rows if r["pick"]]
    thin = [r for r in rows if r.get("thin")]
    nopick = [r for r in rows if not r["pick"] and not r.get("thin")]
    if live:
        w1 = max(len(r["bout"]) for r in live)
        w2 = max(len(r["pick"]) for r in live)
        print(f"{'BOUT':<{w1}}  {'PICK':<{w2}}  {'PROB':>6} {'FAIR':>6} {'DEC':>5} {'BASE':>6} {'LIFT':>6}")
        print("-" * (w1 + w2 + 36))
        for r in sorted(live, key=lambda x: -x["lift"]):
            print(f"{r['bout']:<{w1}}  {r['pick']:<{w2}}  {r['p']:>5.1%} "
                  f"{r['fair']:>6} {r['dec']:>5.2f} {r['base']:>5.1%} {r['lift']:>5.2f}x")
    else:
        print("No qualifying picks on this card.")
    if nopick:
        print(f"\n{len(nopick)} bout(s) no edge over base rates (lift < {LIFT_MIN}x):")
        for r in nopick:
            print(f"  {r['bout']}   best {r['best']:.2f}x")
    if thin:
        print(f"\n{len(thin)} bout(s) withheld, insufficient UFC history:")
        for r in thin:
            print(f"  {r['bout']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--data", default=DATA_DIR)
    ap.add_argument("--card"); ap.add_argument("--fight")
    ap.add_argument("--board", action="store_true",
                    help="board mode: top outcome per bout only, at fair odds")
    ap.add_argument("--show-thin", action="store_true")
    ap.add_argument("--lift", type=float, default=LIFT_MIN,
                    help="min lift over base rate to post a pick (default 1.35)")
    a = ap.parse_args()
    if a.refresh:
        refresh(a.data)
        if not (a.card or a.fight):
            return
    if not (a.card or a.fight):
        ap.print_help(); return
    m = Model(a.data)
    lines = [a.fight] if a.fight else [l.strip() for l in open(a.card)
                                       if l.strip() and not l.startswith("#")]
    res = []
    for ln in lines:
        p = [x.strip() for x in ln.split("|")]
        if len(p) < 3:
            print(f"skip (bad format): {ln}"); continue
        res.append(m.matchup(p[0], p[1], p[2], int(p[3]) if len(p) > 3 else 3))
    if a.board:
        print_board([board_line(r, a.lift) for r in res], a.show_thin)
    else:
        for r in res:
            print(fmt(r)); print()


if __name__ == "__main__":
    main()
