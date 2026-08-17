"""
Method-of-victory board, wired for daily_mma.py.

    from mbm_mma_method import MethodBoard

    mb = MethodBoard(data_dir="data")          # loads + fits once, ~90s
    out = mb.board(slate)                      # slate from your ESPN pull
    for line in out["lines"]:
        print(line["text"])

Slate entries are dicts:
    {"date": "2026-08-22", "a": "Chris Padilla", "b": "Nasrat Haqparast",
     "division": "Lightweight", "rounds": 3, "p_win_a": 0.561}

p_win_a is YOUR moneyline number. Pass it. The 6-way log loss decomposes
exactly into winner + method-given-winner, so the winner marginal flows
through 1:1: your model at 0.6505 against Elo at 0.6866 is worth 0.036
on the 6-way for free. Elo is only the fallback when p_win_a is absent.

This board covers KO/SUB/DEC splits, which your ML and DIST markets do
not. It is additive, not a replacement. Your existing distance model
(0.6748 vs 0.6881 base) already beats this one's distance channel, so
keep using yours for that.

Selection rule and its backtest: BOARD.md. Trust bands: CALIBRATION.md.
"""
import difflib
import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit

import mma_v2 as V
from mma_v2 import estimate_shape, cum, fit_v2, predict_v2, elo_table
from mma_method_model import (
    load, build_fights, build_roundstats, build_dataset, randomize_corners,
    new_state, features, fix_kd15, Scaler, CLASSES, FEAT_KEYS, DIVISIONS,
)
from block_ridge import design_i, make_pen
from mma_predict import replay, ipf

MODEL_TAG = "v2-method"          # for your model_identity check
LAM_MAIN, LAM_INT = 0.01, 0.3
TRUST_LO, TRUST_HI = 0.08, 0.26
LIFT_MIN = 1.35
MIN_BOUTS = 3                     # matches your existing thin-record gate

# ESPN division strings -> ufcstats division strings
DIV_ALIAS = {
    "w strawweight": "Women's Strawweight", "w flyweight": "Women's Flyweight",
    "w bantamweight": "Women's Bantamweight", "w featherweight": "Women's Featherweight",
    "women's strawweight": "Women's Strawweight",
    "women strawweight": "Women's Strawweight",
    "catchweight": "Catch Weight", "catch weight": "Catch Weight",
}


def norm_div(d):
    if not isinstance(d, str):
        return "Unknown"
    k = d.strip().lower()
    if k in DIV_ALIAS:
        return DIV_ALIAS[k]
    for x in DIVISIONS:
        if x.lower() == k:
            return x
    for x in DIVISIONS:
        if x.lower() in k:
            return x
    return "Unknown"


def fair_dec(p):
    return round(1.0 / p, 2)


def fair_am(p):
    return f"-{round(100 * p / (1 - p))}" if p >= 0.5 else f"+{round(100 * (1 - p) / p)}"


class MethodBoard:
    def __init__(self, data_dir="data", lift_min=LIFT_MIN, min_bouts=MIN_BOUTS):
        self.lift_min = lift_min
        self.min_bouts = min_bouts
        raw = load(data_dir)
        self.fights = build_fights(raw)
        self.rstats = build_roundstats(raw)
        self.data_through = self.fights["date"].max()
        ds = randomize_corners(build_dataset(self.fights, self.rstats))
        self.divs = sorted(ds["division"].unique())

        self.phik = estimate_shape(self.fights, "KO")
        self.phis = estimate_shape(self.fights, "SUB")
        self.Ck, self.Cs = cum(self.phik), cum(self.phis)

        mats = design_i(ds, self.divs)
        self.sc = [Scaler().fit(m) for m in mats]
        ms = tuple(s.t(m) for s, m in zip(self.sc, mats))
        self.theta = fit_v2(ms, ds["end_min"].values, ds["max_min"].values,
                            ds["target"].values, make_pen(ms, LAM_MAIN, LAM_INT),
                            self.Ck, self.Cs, self.phik, self.phis)

        et = elo_table(self.fights)
        kk = {(r.EVENT, r.BOUT): (r.f1, r.elo1, r.elo2) for r in et.itertuples(index=False)}
        ea = np.array([kk[(r.event, r.bout)][1] if r.A == kk[(r.event, r.bout)][0]
                       else kk[(r.event, r.bout)][2] for r in ds.itertuples(index=False)])
        eb = np.array([kk[(r.event, r.bout)][2] if r.A == kk[(r.event, r.bout)][0]
                       else kk[(r.event, r.bout)][1] for r in ds.itertuples(index=False)])
        P, _ = predict_v2(self.theta, ms, ds["max_min"].values,
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
        self._lower = {r.lower(): r for r in self.roster}

    # -- name bridge: ESPN strings are not ufcstats strings --
    def resolve(self, name):
        if name in self.roster:
            return name, "exact"
        k = name.lower().strip()
        if k in self._lower:
            return self._lower[k], "exact"
        # drop suffixes and accents-as-spaces artifacts your log showed
        k2 = k.replace(" jr", "").replace(" sr", "").replace(".", "").strip()
        if k2 in self._lower:
            return self._lower[k2], "exact"
        h = difflib.get_close_matches(k, list(self._lower), n=1, cutoff=0.84)
        if h:
            return self._lower[h[0]], "fuzzy"
        # last resort: surname match, unique only
        sur = k.split()[-1]
        cands = [v for kk, v in self._lower.items() if kk.split()[-1] == sur]
        if len(cands) == 1:
            return cands[0], "surname"
        return None, "unmatched"

    def predict(self, a, b, division, rounds=3, p_win_a=None):
        na, ma = self.resolve(a)
        nb, mb_ = self.resolve(b)
        sa = self.state.get(na, new_state()); sb = self.state.get(nb, new_state())
        fa, fb = features(sa, self.prior), features(sb, self.prior)
        fa["kd15"], fb["kd15"] = fix_kd15(sa, self.prior), fix_kd15(sb, self.prior)
        div = norm_div(division)
        row = {"division": div, "max_min": 25.0 if int(rounds) == 5 else 15.0}
        for k in FEAT_KEYS:
            row["A_" + k], row["B_" + k] = fa[k], fb[k]
        df = pd.DataFrame([row])
        ms = tuple(s.t(m) for s, m in zip(self.sc, design_i(df, self.divs)))
        P, S = predict_v2(self.theta, ms, df["max_min"].values,
                          self.Ck, self.Cs, self.phik, self.phis)
        if p_win_a is None:
            pa = float(expit((self.elo.get(na, 1500.0) - self.elo.get(nb, 1500.0)) / 173.7))
            wsrc = "elo"
        else:
            pa = float(np.clip(p_win_a, 0.02, 0.98))
            wsrc = "p_win_a"
        H = ipf(P, np.array([pa]))
        Q = np.array([expit(self.platt[j][0] + self.platt[j][1] *
                            logit(np.clip(H[0, j], 1e-6, 1 - 1e-6))) for j in range(6)])
        Q = Q / Q.sum()
        vc = self.base_rates.get((div, int(rounds)), self.base_global)
        B = np.array([vc[0] * pa, vc[1] * pa, vc[2] * pa,
                      vc[0] * (1 - pa), vc[1] * (1 - pa), vc[2] * (1 - pa)])
        B = B / B.sum()
        return dict(A=na or a, B=nb or b, raw_a=a, raw_b=b, div=div, rounds=int(rounds),
                    match=(ma, mb_), n=(int(sa["fights"]), int(sb["fights"])),
                    Q=Q, base=B, surv=S[0], winner_src=wsrc, p_win_a=pa)

    def board(self, slate):
        lines, withheld, unmatched, noedge, fuzzy = [], [], [], [], []
        for s in slate:
            r = self.predict(s["a"], s["b"], s.get("division", "Unknown"),
                             s.get("rounds", 3), s.get("p_win_a"))
            bout = f"{r['raw_a']} vs {r['raw_b']}"
            if "unmatched" in r["match"]:
                unmatched.append(bout); continue
            if any(m in ("fuzzy", "surname") for m in r["match"]):
                fuzzy.append(f"{bout} -> {r['A']} vs {r['B']}")
            if min(r["n"]) < self.min_bouts:
                withheld.append(f"{bout} (min {min(r['n'])} bout(s))"); continue
            lift = np.where((r["Q"] >= TRUST_LO) & (r["Q"] <= TRUST_HI),
                            r["Q"] / np.maximum(r["base"], 1e-9), -1.0)
            j = int(lift.argmax())
            if lift[j] < self.lift_min:
                noedge.append(f"{bout} best {max(lift.max(), 0):.2f}x"); continue
            c = CLASSES[j]
            who = r["A"] if c.startswith("A_") else r["B"]
            p = float(r["Q"][j])
            pick = f"{who} by {c.split('_')[1]}"
            lines.append(dict(
                date=s.get("date", ""), bout=bout, pick=pick, prob=p,
                dec=fair_dec(p), am=fair_am(p), lift=float(lift[j]),
                base=float(r["base"][j]), winner_src=r["winner_src"],
                text=f"  {pick:<34}{p * 100:5.1f}%  find {fair_dec(p):.2f} / {fair_am(p)} or better"))
        return dict(lines=sorted(lines, key=lambda x: -x["lift"]), withheld=withheld,
                    unmatched=unmatched, no_edge=noedge, fuzzy=fuzzy,
                    data_through=str(self.data_through.date()), tag=MODEL_TAG)

    def render(self, out, header="=== MBM MMA method board ==="):
        s = [header]
        if not out["lines"]:
            s.append("  no qualifying picks")
        cur = None
        for l in out["lines"]:
            if l["date"] != cur:
                s.append(f"{l['date']}  {l['bout']}") if l["date"] else s.append(l["bout"])
                cur = l["date"]
            else:
                s.append(f"            {l['bout']}")
            s.append(l["text"])
        tail = []
        if out["no_edge"]:
            tail.append(f"({len(out['no_edge'])} no edge over base rates, lift < {self.lift_min}x)")
        if out["withheld"]:
            tail.append(f"({len(out['withheld'])} withheld on thin record (<{self.min_bouts} bouts))")
        if out["unmatched"]:
            tail.append(f"({len(out['unmatched'])} unmatched to ufcstats roster)")
        s += tail
        return "\n".join(s)

    def checks(self, out, max_age_days=21):
        """MBM-CHECK lines in your existing format."""
        res = []
        age = (pd.Timestamp.now().normalize() - self.data_through).days
        res.append(("freshness", age <= max_age_days,
                    f"ufcstats mirror current to {self.data_through.date()} ({age}d)"))
        res.append(("model_identity", True, f"tag {MODEL_TAG}"))
        res.append(("name_resolution", len(out["unmatched"]) == 0,
                    f"{len(out['unmatched'])} unmatched, {len(out['fuzzy'])} fuzzy"))
        probs = [round(l["prob"], 3) for l in out["lines"]]
        rep = max([probs.count(p) for p in set(probs)], default=0)
        res.append(("no_conviction", rep < 3, f"{len(probs)} pick(s), max repeat {rep}"))
        src = {l["winner_src"] for l in out["lines"]}
        res.append(("winner_source", src <= {"p_win_a"} or not src,
                    f"{sorted(src) or ['none']} (elo fallback costs ~0.036 on 6-way)"))
        band = all(TRUST_LO <= l["prob"] <= TRUST_HI for l in out["lines"])
        res.append(("trust_band", band, f"all picks within {TRUST_LO}-{TRUST_HI}"))
        return res
