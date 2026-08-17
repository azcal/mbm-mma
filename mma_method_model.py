"""
MMA method-of-victory model.

Predicts a 6-way distribution over {A-KO, A-SUB, A-DEC, B-KO, B-SUB, B-DEC}
using a competing-hazards formulation. Odds-blind. No market inputs.

Data source: Greco1899/scrape_ufc_stats maintained CSV mirror of UFC Stats.

Core idea:
    lambda_ko(A->B) = exp(w . [A's finishing ability, B's vulnerability, division])
    lambda_sub(A->B) = same shape
    Fight survives to decision iff no hazard fires within the scheduled time.

All fighter features are point-in-time (career-to-date, strictly before the
fight being predicted) with empirical-Bayes shrinkage toward a running
division prior. No future information enters any feature.
"""

import io
import re
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

BASE = "https://raw.githubusercontent.com/Greco1899/scrape_ufc_stats/main"

DIVISIONS = [
    "Women's Strawweight", "Women's Flyweight", "Women's Bantamweight",
    "Women's Featherweight", "Strawweight", "Flyweight", "Bantamweight",
    "Featherweight", "Lightweight", "Welterweight", "Middleweight",
    "Light Heavyweight", "Heavyweight", "Catch Weight", "Open Weight",
]

# Scheduled formats we trust. Everything else is legacy/no-limit weirdness.
FORMATS = {"3 Rnd (5-5-5)": 900.0, "5 Rnd (5-5-5-5-5)": 1500.0}


# ----------------------------------------------------------------------
# Loading and parsing
# ----------------------------------------------------------------------

def load(local_dir=None):
    names = ["ufc_event_details", "ufc_fight_results", "ufc_fight_stats"]
    out = {}
    for n in names:
        if local_dir:
            out[n] = pd.read_csv(f"{local_dir}/{n}.csv")
        else:
            out[n] = pd.read_csv(f"{BASE}/{n}.csv")
    return out


def parse_of(s):
    """'3 of 5' -> (3, 5)"""
    if not isinstance(s, str) or " of " not in s:
        return (np.nan, np.nan)
    a, b = s.split(" of ")
    try:
        return (float(a), float(b))
    except ValueError:
        return (np.nan, np.nan)


def parse_clock(s):
    """'3:38' -> 218.0 seconds"""
    if not isinstance(s, str) or ":" not in s:
        return np.nan
    m, sec = s.split(":")[:2]
    try:
        return float(m) * 60 + float(sec)
    except ValueError:
        return np.nan


def extract_division(wc):
    if not isinstance(wc, str):
        return "Unknown"
    for d in DIVISIONS:
        if d.lower() in wc.lower():
            return d
    return "Unknown"


def classify_method(m):
    if not isinstance(m, str):
        return None
    m = m.strip()
    if m.startswith("KO/TKO") or m.startswith("TKO"):
        return "KO"
    if m.startswith("Submission"):
        return "SUB"
    if m.startswith("Decision"):
        return "DEC"
    return None  # DQ, Overturned, Could Not Continue, Other


def build_fights(raw):
    ev = raw["ufc_event_details"].copy()
    ev["EVENT"] = ev["EVENT"].str.strip()
    ev["DATE"] = pd.to_datetime(ev["DATE"], format="mixed")
    date_map = dict(zip(ev["EVENT"], ev["DATE"]))

    r = raw["ufc_fight_results"].copy()
    r["EVENT"] = r["EVENT"].str.strip()
    r["BOUT"] = r["BOUT"].str.strip()
    r["date"] = r["EVENT"].map(date_map)

    r["method"] = r["METHOD"].apply(classify_method)
    r["division"] = r["WEIGHTCLASS"].apply(extract_division)
    r["fmt"] = r["TIME FORMAT"].str.strip()
    r["max_sec"] = r["fmt"].map(FORMATS)

    parts = r["BOUT"].str.split(" vs. ", n=1, expand=True)
    r["f1"] = parts[0].str.strip()
    r["f2"] = parts[1].str.strip()

    out = r["OUTCOME"].str.strip()
    r["winner_idx"] = np.where(out == "W/L", 1, np.where(out == "L/W", 2, 0))

    r["end_sec"] = (r["ROUND"] - 1) * 300.0 + r["TIME"].apply(parse_clock)

    keep = (
        r["method"].notna()
        & r["max_sec"].notna()
        & (r["winner_idx"] > 0)
        & r["date"].notna()
        & r["end_sec"].notna()
        & r["f1"].notna()
        & r["f2"].notna()
    )
    r = r[keep].copy()

    # Decisions clock out at the scheduled distance.
    r.loc[r["method"] == "DEC", "end_sec"] = r.loc[r["method"] == "DEC", "max_sec"]
    r["end_sec"] = r[["end_sec", "max_sec"]].min(axis=1)
    r = r[r["end_sec"] > 0]

    r["winner"] = np.where(r["winner_idx"] == 1, r["f1"], r["f2"])
    r["loser"] = np.where(r["winner_idx"] == 1, r["f2"], r["f1"])

    return r.sort_values("date").reset_index(drop=True)


def build_roundstats(raw):
    s = raw["ufc_fight_stats"].copy()
    s["EVENT"] = s["EVENT"].str.strip()
    s["BOUT"] = s["BOUT"].str.strip()
    s["FIGHTER"] = s["FIGHTER"].str.strip()

    for col, pre in [("SIG.STR.", "sig"), ("TOTAL STR.", "tot"), ("TD", "td")]:
        parsed = s[col].apply(parse_of)
        s[f"{pre}_l"] = [p[0] for p in parsed]
        s[f"{pre}_a"] = [p[1] for p in parsed]

    s["ctrl"] = s["CTRL"].apply(parse_clock).fillna(0.0)
    s["kd"] = pd.to_numeric(s["KD"], errors="coerce").fillna(0.0)
    s["subatt"] = pd.to_numeric(s["SUB.ATT"], errors="coerce").fillna(0.0)

    agg = (
        s.groupby(["EVENT", "BOUT", "FIGHTER"], as_index=False)
        .agg(
            kd=("kd", "sum"),
            subatt=("subatt", "sum"),
            ctrl=("ctrl", "sum"),
            sig_l=("sig_l", "sum"),
            sig_a=("sig_a", "sum"),
            td_l=("td_l", "sum"),
            td_a=("td_a", "sum"),
        )
    )
    return agg


# ----------------------------------------------------------------------
# Point-in-time career state
# ----------------------------------------------------------------------

CTR_KEYS = [
    "mins", "fights",
    "kd_for", "kd_against",
    "sig_l", "sig_a", "sig_l_opp", "sig_a_opp",
    "td_l", "td_a", "td_l_opp", "td_a_opp",
    "sub_for", "sub_against",
    "ctrl_for", "ctrl_against",
    "w_ko", "w_sub", "w_dec", "l_ko", "l_sub", "l_dec",
]


def new_state():
    return {k: 0.0 for k in CTR_KEYS}


class RunningPrior:
    """Expanding global rates. Used as the shrinkage target so nothing leaks."""

    def __init__(self):
        self.n = 0.0
        self.acc = {}

    def get(self, key, fallback):
        if self.n < 50:
            return fallback
        return self.acc.get(key, 0.0) / self.n

    def add(self, d):
        self.n += 1
        for k, v in d.items():
            self.acc[k] = self.acc.get(k, 0.0) + v


def shrink_rate(num, den, prior, k):
    """Empirical-Bayes shrinkage for a rate. k is pseudo-exposure."""
    return (num + k * prior) / (den + k)


def features(st, prior):
    """Career-to-date features for one fighter, shrunk toward running priors."""
    mins = st["mins"]
    f = st["fights"]

    p = {
        "kd15": prior.get("kd15", 0.55),
        "kda15": prior.get("kda15", 0.55),
        "slpm": prior.get("slpm", 3.6),
        "sapm": prior.get("sapm", 3.6),
        "acc": prior.get("acc", 0.44),
        "dfn": prior.get("dfn", 0.56),
        "td15": prior.get("td15", 1.6),
        "tdacc": prior.get("tdacc", 0.36),
        "tddef": prior.get("tddef", 0.64),
        "sub15": prior.get("sub15", 0.55),
        "suba15": prior.get("suba15", 0.55),
        "ctrl": prior.get("ctrl", 0.12),
        "wko": prior.get("wko", 0.19),
        "wsub": prior.get("wsub", 0.12),
        "lko": prior.get("lko", 0.19),
        "lsub": prior.get("lsub", 0.12),
    }

    KM = 25.0   # pseudo-minutes, ~2 fights
    KF = 4.0    # pseudo-fights for proportions

    return {
        "kd15": shrink_rate(st["kd_for"] * 15, mins, p["kd15"] * KM, KM) if True else 0,
        "kda15": shrink_rate(st["kd_against"] * 15, mins, p["kda15"] * KM, KM),
        "slpm": shrink_rate(st["sig_l"], mins, p["slpm"] * KM, KM),
        "sapm": shrink_rate(st["sig_l_opp"], mins, p["sapm"] * KM, KM),
        "acc": shrink_rate(st["sig_l"], st["sig_a"], p["acc"] * 60, 60),
        "dfn": shrink_rate(st["sig_a_opp"] - st["sig_l_opp"], st["sig_a_opp"], p["dfn"] * 60, 60),
        "td15": shrink_rate(st["td_l"] * 15, mins, p["td15"] * KM, KM),
        "tdacc": shrink_rate(st["td_l"], st["td_a"], p["tdacc"] * 8, 8),
        "tddef": shrink_rate(st["td_a_opp"] - st["td_l_opp"], st["td_a_opp"], p["tddef"] * 8, 8),
        "sub15": shrink_rate(st["sub_for"] * 15, mins, p["sub15"] * KM, KM),
        "suba15": shrink_rate(st["sub_against"] * 15, mins, p["suba15"] * KM, KM),
        "ctrl": shrink_rate(st["ctrl_for"], mins * 60, p["ctrl"] * KM * 60, KM * 60),
        "wko": shrink_rate(st["w_ko"], f, p["wko"] * KF, KF),
        "wsub": shrink_rate(st["w_sub"], f, p["wsub"] * KF, KF),
        "lko": shrink_rate(st["l_ko"], f, p["lko"] * KF, KF),
        "lsub": shrink_rate(st["l_sub"], f, p["lsub"] * KF, KF),
        "exp": np.log1p(f),
    }


def fix_kd15(st, prior):
    KM = 25.0
    return shrink_rate(st["kd_for"] * 15, st["mins"], prior.get("kd15", 0.55) * KM, KM)


def build_dataset(fights, rstats):
    """One row per fight with point-in-time features for both corners."""
    idx = {}
    for rec in rstats.itertuples(index=False):
        idx[(rec.EVENT, rec.BOUT, rec.FIGHTER)] = rec

    state = {}
    prior = RunningPrior()
    rows = []

    for fr in fights.itertuples(index=False):
        a, b = fr.f1, fr.f2
        sa = state.setdefault(a, new_state())
        sb = state.setdefault(b, new_state())

        fa = features(sa, prior)
        fb = features(sb, prior)
        fa["kd15"] = fix_kd15(sa, prior)
        fb["kd15"] = fix_kd15(sb, prior)

        # Target: 6-way outcome from A's perspective.
        a_won = fr.winner == a
        tgt = ("A" if a_won else "B") + "_" + fr.method

        row = {
            "date": fr.date, "event": fr.EVENT, "bout": fr.BOUT,
            "A": a, "B": b, "division": fr.division,
            "max_min": fr.max_sec / 60.0, "end_min": fr.end_sec / 60.0,
            "method": fr.method, "a_won": a_won, "target": tgt,
            "n_prior_A": sa["fights"], "n_prior_B": sb["fights"],
        }
        for k, v in fa.items():
            row["A_" + k] = v
        for k, v in fb.items():
            row["B_" + k] = v
        rows.append(row)

        # ---- update state AFTER emitting the row (point-in-time guarantee) ----
        ra = idx.get((fr.EVENT, fr.BOUT, a))
        rb = idx.get((fr.EVENT, fr.BOUT, b))
        mins = fr.end_sec / 60.0

        for me, opp, st_me, st_opp in ((ra, rb, sa, sb), (rb, ra, sb, sa)):
            st_me["mins"] += mins
            st_me["fights"] += 1
            if me is not None:
                st_me["kd_for"] += me.kd
                st_me["sig_l"] += me.sig_l
                st_me["sig_a"] += me.sig_a
                st_me["td_l"] += me.td_l
                st_me["td_a"] += me.td_a
                st_me["sub_for"] += me.subatt
                st_me["ctrl_for"] += me.ctrl
            if opp is not None:
                st_me["kd_against"] += opp.kd
                st_me["sig_l_opp"] += opp.sig_l
                st_me["sig_a_opp"] += opp.sig_a
                st_me["td_l_opp"] += opp.td_l
                st_me["td_a_opp"] += opp.td_a
                st_me["sub_against"] += opp.subatt
                st_me["ctrl_against"] += opp.ctrl

        wst = sa if a_won else sb
        lst = sb if a_won else sa
        wst["w_" + fr.method.lower()] += 1
        lst["l_" + fr.method.lower()] += 1

        if mins > 0 and ra is not None and rb is not None:
            prior.add({
                "kd15": (ra.kd + rb.kd) / 2 * 15 / mins,
                "kda15": (ra.kd + rb.kd) / 2 * 15 / mins,
                "slpm": (ra.sig_l + rb.sig_l) / 2 / mins,
                "sapm": (ra.sig_l + rb.sig_l) / 2 / mins,
                "acc": np.nan_to_num((ra.sig_l + rb.sig_l) / max(ra.sig_a + rb.sig_a, 1)),
                "dfn": 1 - np.nan_to_num((ra.sig_l + rb.sig_l) / max(ra.sig_a + rb.sig_a, 1)),
                "td15": (ra.td_l + rb.td_l) / 2 * 15 / mins,
                "tdacc": np.nan_to_num((ra.td_l + rb.td_l) / max(ra.td_a + rb.td_a, 1)),
                "tddef": 1 - np.nan_to_num((ra.td_l + rb.td_l) / max(ra.td_a + rb.td_a, 1)),
                "sub15": (ra.subatt + rb.subatt) / 2 * 15 / mins,
                "suba15": (ra.subatt + rb.subatt) / 2 * 15 / mins,
                "ctrl": (ra.ctrl + rb.ctrl) / 2 / (mins * 60),
                "wko": 1.0 if fr.method == "KO" else 0.0,
                "wsub": 1.0 if fr.method == "SUB" else 0.0,
                "lko": 1.0 if fr.method == "KO" else 0.0,
                "lsub": 1.0 if fr.method == "SUB" else 0.0,
            })

    return pd.DataFrame(rows)


FEAT_KEYS = ["kd15", "kda15", "slpm", "sapm", "acc", "dfn", "td15", "tdacc",
             "tddef", "sub15", "suba15", "ctrl", "wko", "wsub", "lko", "lsub", "exp"]


def randomize_corners(ds, seed=7):
    """
    UFC Stats lists the winner first in ~64% of bouts. Left alone, any model
    with a corner intercept learns that artifact instead of learning fighting.
    Coin-flip the corner assignment so P(A wins) is 50% by construction.
    """
    rng = np.random.default_rng(seed)
    flip = rng.random(len(ds)) < 0.5
    out = ds.copy()

    out.loc[flip, "A"], out.loc[flip, "B"] = ds.loc[flip, "B"].values, ds.loc[flip, "A"].values
    out.loc[flip, "n_prior_A"], out.loc[flip, "n_prior_B"] = (
        ds.loc[flip, "n_prior_B"].values, ds.loc[flip, "n_prior_A"].values)
    for k in FEAT_KEYS:
        out.loc[flip, "A_" + k] = ds.loc[flip, "B_" + k].values
        out.loc[flip, "B_" + k] = ds.loc[flip, "A_" + k].values

    out.loc[flip, "a_won"] = ~ds.loc[flip, "a_won"].astype(bool)
    out["target"] = np.where(out["a_won"], "A_", "B_") + out["method"]
    return out


# ----------------------------------------------------------------------
# Competing-hazards model
# ----------------------------------------------------------------------

KO_ATK = ["kd15", "slpm", "acc", "wko", "td15"]
KO_DEF = ["kda15", "lko", "dfn", "sapm", "tddef"]
SUB_ATK = ["sub15", "wsub", "td15", "tdacc", "ctrl"]
SUB_DEF = ["lsub", "tddef", "suba15", "dfn"]
DEC_F = ["slpm", "acc", "dfn", "td15", "ctrl", "sapm", "exp"]

CLASSES = ["A_KO", "A_SUB", "A_DEC", "B_KO", "B_SUB", "B_DEC"]


def design(df, divs):
    """Build the four hazard design matrices plus the decision matrix."""
    def blk(atk, dfn, atk_p, dfn_p):
        cols = [df[atk_p + c].values for c in atk] + [df[dfn_p + c].values for c in dfn]
        return np.column_stack(cols)

    D = np.column_stack([(df["division"] == d).astype(float).values for d in divs])
    five = (df["max_min"] > 20).astype(float).values.reshape(-1, 1)

    Xko_ab = np.hstack([blk(KO_ATK, KO_DEF, "A_", "B_"), D, five])
    Xko_ba = np.hstack([blk(KO_ATK, KO_DEF, "B_", "A_"), D, five])
    Xsb_ab = np.hstack([blk(SUB_ATK, SUB_DEF, "A_", "B_"), D, five])
    Xsb_ba = np.hstack([blk(SUB_ATK, SUB_DEF, "B_", "A_"), D, five])

    Xdec = np.column_stack(
        [df["A_" + c].values - df["B_" + c].values for c in DEC_F]
    )
    return Xko_ab, Xko_ba, Xsb_ab, Xsb_ba, Xdec


class Scaler:
    def fit(self, X):
        self.m = X.mean(0)
        self.s = X.std(0)
        self.s[self.s < 1e-9] = 1.0
        return self

    def t(self, X):
        return (X - self.m) / self.s


def _unpack(theta, pk, ps, pd_):
    i = 0
    bk = theta[i:i + pk]; i += pk
    ak = theta[i]; i += 1
    bs = theta[i:i + ps]; i += ps
    as_ = theta[i]; i += 1
    bd = theta[i:i + pd_]; i += pd_
    ad = theta[i]
    return bk, ak, bs, as_, bd, ad


def nll_grad(theta, mats, t, T, y, lam):
    """Negative mean log-likelihood and its analytic gradient."""
    Xk1, Xk2, Xs1, Xs2, Xd = mats
    pk, ps, pd_ = Xk1.shape[1], Xs1.shape[1], Xd.shape[1]
    bk, ak, bs, as_, bd, ad = _unpack(theta, pk, ps, pd_)
    n = len(t)

    l_k1 = np.exp(np.clip(Xk1 @ bk + ak, -30, 6))
    l_k2 = np.exp(np.clip(Xk2 @ bk + ak, -30, 6))
    l_s1 = np.exp(np.clip(Xs1 @ bs + as_, -30, 6))
    l_s2 = np.exp(np.clip(Xs2 @ bs + as_, -30, 6))
    L = l_k1 + l_k2 + l_s1 + l_s2

    is_dec = (y == "A_DEC") | (y == "B_DEC")
    u = np.where(is_dec, T, t)

    i_k1 = (y == "A_KO").astype(float)
    i_k2 = (y == "B_KO").astype(float)
    i_s1 = (y == "A_SUB").astype(float)
    i_s2 = (y == "B_SUB").astype(float)

    ll = -L * u
    ll += i_k1 * np.log(l_k1 + 1e-300)
    ll += i_k2 * np.log(l_k2 + 1e-300)
    ll += i_s1 * np.log(l_s1 + 1e-300)
    ll += i_s2 * np.log(l_s2 + 1e-300)

    z = np.clip(Xd @ bd + ad, -30, 30)
    p_a = expit(z)
    a_dec = (y == "A_DEC")
    b_dec = (y == "B_DEC")
    ll += a_dec * np.log(p_a + 1e-12)
    ll += b_dec * np.log(1 - p_a + 1e-12)

    f = -ll.sum() / n + lam * (bk @ bk + bs @ bs + bd @ bd)

    # ---- gradient ----
    w_k1 = i_k1 - u * l_k1
    w_k2 = i_k2 - u * l_k2
    w_s1 = i_s1 - u * l_s1
    w_s2 = i_s2 - u * l_s2

    g_bk = -(Xk1.T @ w_k1 + Xk2.T @ w_k2) / n + 2 * lam * bk
    g_ak = -(w_k1.sum() + w_k2.sum()) / n
    g_bs = -(Xs1.T @ w_s1 + Xs2.T @ w_s2) / n + 2 * lam * bs
    g_as = -(w_s1.sum() + w_s2.sum()) / n

    gd = a_dec * (1 - p_a) - b_dec * p_a
    g_bd = -(Xd.T @ gd) / n + 2 * lam * bd
    g_ad = -gd.sum() / n

    grad = np.concatenate([g_bk, [g_ak], g_bs, [g_as], g_bd, [g_ad]])
    return f, grad


def nll(theta, mats, t, T, y, lam):
    return nll_grad(theta, mats, t, T, y, lam)[0]


def n_params(mats):
    """bk and bs are shared across both directions, so count each block once."""
    return mats[0].shape[1] + mats[2].shape[1] + mats[4].shape[1] + 3


def fit(mats, t, T, y, lam):
    pk, ps = mats[0].shape[1], mats[2].shape[1]
    x0 = np.zeros(n_params(mats))
    x0[pk] = np.log(0.02)            # KO intercept: ~0.02 finishes/min
    x0[pk + 1 + ps] = np.log(0.012)  # SUB intercept
    res = minimize(nll_grad, x0, args=(mats, t, T, y, lam), jac=True,
                   method="L-BFGS-B", options={"maxiter": 2000, "ftol": 1e-12})
    return res.x


def predict(theta, mats, T):
    Xk1, Xk2, Xs1, Xs2, Xd = mats
    pk, ps, pd_ = Xk1.shape[1], Xs1.shape[1], Xd.shape[1]
    i = 0
    bk = theta[i:i + pk]; i += pk
    ak_k = theta[i]; i += 1
    bs = theta[i:i + ps]; i += ps
    ak_s = theta[i]; i += 1
    bd = theta[i:i + pd_]; i += pd_
    ad = theta[i]

    l = {
        "A_KO": np.exp(np.clip(Xk1 @ bk + ak_k, -30, 6)),
        "B_KO": np.exp(np.clip(Xk2 @ bk + ak_k, -30, 6)),
        "A_SUB": np.exp(np.clip(Xs1 @ bs + ak_s, -30, 6)),
        "B_SUB": np.exp(np.clip(Xs2 @ bs + ak_s, -30, 6)),
    }
    L = sum(l.values())
    p_dec = np.exp(-L * T)
    p_fin = 1 - p_dec
    p_a = expit(np.clip(Xd @ bd + ad, -30, 30))

    P = np.column_stack([
        l["A_KO"] / L * p_fin,
        l["A_SUB"] / L * p_fin,
        p_dec * p_a,
        l["B_KO"] / L * p_fin,
        l["B_SUB"] / L * p_fin,
        p_dec * (1 - p_a),
    ])
    return P / P.sum(1, keepdims=True)


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------

def logloss(P, y):
    yi = np.array([CLASSES.index(v) for v in y])
    return -np.log(np.clip(P[np.arange(len(yi)), yi], 1e-15, 1)).mean()


def brier(P, y):
    Y = np.zeros_like(P)
    for i, v in enumerate(y):
        Y[i, CLASSES.index(v)] = 1.0
    return ((P - Y) ** 2).sum(1).mean()


def baseline(train, test):
    """Division + round-format method base rates, 50/50 on the winner."""
    P = np.zeros((len(test), 6))
    glob = train["method"].value_counts(normalize=True)
    for i, r in enumerate(test.itertuples(index=False)):
        sub = train[(train["division"] == r.division) &
                    ((train["max_min"] > 20) == (r.max_min > 20))]
        vc = sub["method"].value_counts(normalize=True) if len(sub) >= 40 else glob
        ko, sb = vc.get("KO", 0.32), vc.get("SUB", 0.19)
        dc = max(1 - ko - sb, 1e-6)
        P[i] = [ko / 2, sb / 2, dc / 2, ko / 2, sb / 2, dc / 2]
    return P / P.sum(1, keepdims=True)


def walk_forward(ds, start_year=2015, lam=0.02):
    divs = sorted(ds["division"].unique())
    years = sorted(y for y in ds["date"].dt.year.unique() if y >= start_year)
    res = []
    for yr in years:
        tr = ds[ds["date"].dt.year < yr]
        te = ds[ds["date"].dt.year == yr]
        if len(tr) < 800 or len(te) < 30:
            continue

        mtr = design(tr, divs)
        mte = design(te, divs)
        sc = [Scaler().fit(m) for m in mtr]
        mtr = tuple(s.t(m) for s, m in zip(sc, mtr))
        mte = tuple(s.t(m) for s, m in zip(sc, mte))

        th = fit(mtr, tr["end_min"].values, tr["max_min"].values,
                 tr["target"].values, lam)
        P = predict(th, mte, te["max_min"].values)
        B = baseline(tr, te)
        y = te["target"].values

        res.append({
            "year": yr, "n": len(te),
            "model_ll": logloss(P, y), "base_ll": logloss(B, y),
            "model_br": brier(P, y), "base_br": brier(B, y),
        })
    return pd.DataFrame(res)


if __name__ == "__main__":
    local = sys.argv[1] if len(sys.argv) > 1 else None
    raw = load(local)
    fights = build_fights(raw)
    rstats = build_roundstats(raw)
    print(f"fights parsed: {len(fights)}  ({fights['date'].min().date()} -> {fights['date'].max().date()})")
    ds = build_dataset(fights, rstats)
    ds = randomize_corners(ds)
    ds.to_csv("/home/claude/mma_dataset.csv", index=False)
    print(f"dataset rows: {len(ds)}")
    print(ds["target"].value_counts(normalize=True).round(4).to_string())
    r = walk_forward(ds)
    print()
    print(r.round(4).to_string(index=False))
    w = r["n"] / r["n"].sum()
    print()
    print(f"weighted model log loss : {(r['model_ll'] * w).sum():.4f}")
    print(f"weighted base  log loss : {(r['base_ll'] * w).sum():.4f}")
    print(f"delta                   : {(r['model_ll'] * w).sum() - (r['base_ll'] * w).sum():+.4f}")
