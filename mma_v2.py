"""
MMA v2: proportional hazards with a non-constant baseline shape.

Three upgrades over v1:

1. BASELINE HAZARD SHAPE. v1 assumed a flat finishing hazard. It is not.
   Round 3 KO hazard is under half of round 1, and every round has an
   end-of-round spike and an opening lull. v2 estimates a shared shape
   phi(t) non-parametrically and models lambda_i(t) = lambda_i * phi(t).
   This unlocks round totals and distance markets, which a terminal
   multinomial cannot price at all.

2. PROPER BENCHMARK LADDER. Baselines must include a winner model, or the
   6-way comparison hands the model free credit for picking winners.
   Ladder runs uniform -> global -> division base rates -> division base
   rates x Elo.

3. MARKET COLLAPSES. The 6-way is the hardest form of the bet. Books also
   offer 4-way (inside distance / decision, by fighter), 3-way method,
   goes-the-distance, and round totals. Edge is evaluated separately on
   each, because it does not distribute evenly.

Plus Monte Carlo over PARAMETER uncertainty, which is where simulation
actually earns its keep with one observation per fight.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

import mma_method_model as M
from mma_method_model import (
    load, build_fights, build_roundstats, build_dataset, randomize_corners,
    Scaler, CLASSES,
)
from block_ridge import design_i, make_pen, N_INT

GRID = 0.25          # minutes per integration step
MAXT = 25.0
NG = int(MAXT / GRID)
TGRID = (np.arange(NG) + 0.5) * GRID     # bin midpoints


# ----------------------------------------------------------------------
# 1. Baseline hazard shape
# ----------------------------------------------------------------------

def estimate_shape(fights, method, smooth=0.35):
    """
    Non-parametric baseline hazard shape on the time grid, normalised to
    mean 1. Uses correct risk sets: a fight is at risk at time t only if
    its scheduled distance extends past t and it had not yet ended.

    Decomposed as round multiplier x within-round shape so the sparse
    championship rounds borrow strength from rounds 1-3.
    """
    end = fights["end_sec"].values / 60.0
    mx = fights["max_sec"].values / 60.0
    is_m = (fights["method"].values == method)

    ev = np.zeros(NG)
    ar = np.zeros(NG)
    for g in range(NG):
        lo, hi = g * GRID, (g + 1) * GRID
        at_risk = (mx > lo) & (end > lo)
        ar[g] = at_risk.sum()
        ev[g] = (is_m & at_risk & (end <= hi)).sum()

    rnd = np.minimum((TGRID // 5).astype(int), 4)      # 0..4
    win = ((TGRID % 5) // (5 / (5 / GRID / 5))).astype(int) if False else \
        ((TGRID % 5) / GRID).astype(int)               # step within round

    # Iterative proportional fitting of h ~ R[round] * W[within]
    ok = ar > 30
    R = np.ones(5)
    W = np.ones(int(5 / GRID))
    for _ in range(60):
        for r in range(5):
            m = ok & (rnd == r)
            den = (ar[m] * W[win[m]]).sum()
            R[r] = ev[m].sum() / den if den > 0 else R[r]
        for w in range(len(W)):
            m = ok & (win == w)
            den = (ar[m] * R[rnd[m]]).sum()
            W[w] = ev[m].sum() / den if den > 0 else W[w]

    phi = R[rnd] * W[win]
    # shrink toward flat to keep thin bins from going wild
    phi = smooth * phi + (1 - smooth) * phi.mean()
    phi = np.clip(phi, 1e-4, None)
    phi = phi / phi.mean()
    return phi


def cum(phi):
    """Cumulative integral of phi on the grid: Phi(t) at bin edges."""
    return np.concatenate([[0.0], np.cumsum(phi) * GRID])


def phi_at(phi, t):
    """phi evaluated at times t."""
    g = np.clip((np.asarray(t) / GRID).astype(int), 0, NG - 1)
    return phi[g]


def Phi_at(C, t):
    """Cumulative hazard Phi(t), linearly interpolated."""
    t = np.asarray(t, dtype=float)
    g = np.clip((t / GRID).astype(int), 0, NG - 1)
    return C[g] + (t - g * GRID) * (C[g + 1] - C[g]) / GRID


# ----------------------------------------------------------------------
# 2. Likelihood with time-varying baseline
# ----------------------------------------------------------------------

def nll_v2(theta, mats, t, T, y, pen, Cko, Csub, phik, phis):
    Xk1, Xk2, Xs1, Xs2, Xd = mats
    pk, ps, pd_ = Xk1.shape[1], Xs1.shape[1], Xd.shape[1]
    bk, ak, bs, as_, bd, ad = M._unpack(theta, pk, ps, pd_)
    n = len(t)

    l_k1 = np.exp(np.clip(Xk1 @ bk + ak, -30, 6))
    l_k2 = np.exp(np.clip(Xk2 @ bk + ak, -30, 6))
    l_s1 = np.exp(np.clip(Xs1 @ bs + as_, -30, 6))
    l_s2 = np.exp(np.clip(Xs2 @ bs + as_, -30, 6))

    is_dec = (y == "A_DEC") | (y == "B_DEC")
    u = np.where(is_dec, T, t)
    CK = Phi_at(Cko, u)      # integrated KO baseline up to u
    CS = Phi_at(Csub, u)

    i_k1 = (y == "A_KO").astype(float); i_k2 = (y == "B_KO").astype(float)
    i_s1 = (y == "A_SUB").astype(float); i_s2 = (y == "B_SUB").astype(float)

    lg_k = np.log(phi_at(phik, u) + 1e-300)
    lg_s = np.log(phi_at(phis, u) + 1e-300)

    ll = -(l_k1 + l_k2) * CK - (l_s1 + l_s2) * CS
    ll += i_k1 * (np.log(l_k1 + 1e-300) + lg_k)
    ll += i_k2 * (np.log(l_k2 + 1e-300) + lg_k)
    ll += i_s1 * (np.log(l_s1 + 1e-300) + lg_s)
    ll += i_s2 * (np.log(l_s2 + 1e-300) + lg_s)

    p_a = expit(np.clip(Xd @ bd + ad, -30, 30))
    a_d = (y == "A_DEC"); b_d = (y == "B_DEC")
    ll += a_d * np.log(p_a + 1e-12) + b_d * np.log(1 - p_a + 1e-12)

    f = -ll.sum() / n + float(pen @ (theta ** 2))

    w_k1 = i_k1 - CK * l_k1; w_k2 = i_k2 - CK * l_k2
    w_s1 = i_s1 - CS * l_s1; w_s2 = i_s2 - CS * l_s2
    gd = a_d * (1 - p_a) - b_d * p_a

    g = np.concatenate([
        -(Xk1.T @ w_k1 + Xk2.T @ w_k2) / n, [-(w_k1.sum() + w_k2.sum()) / n],
        -(Xs1.T @ w_s1 + Xs2.T @ w_s2) / n, [-(w_s1.sum() + w_s2.sum()) / n],
        -(Xd.T @ gd) / n, [-gd.sum() / n],
    ]) + 2 * pen * theta
    return f, g


def fit_v2(mats, t, T, y, pen, Cko, Csub, phik, phis):
    pk, ps = mats[0].shape[1], mats[2].shape[1]
    x0 = np.zeros(M.n_params(mats))
    x0[pk] = np.log(0.02); x0[pk + 1 + ps] = np.log(0.012)
    return minimize(nll_v2, x0,
                    args=(mats, t, T, y, pen, Cko, Csub, phik, phis),
                    jac=True, method="L-BFGS-B",
                    options={"maxiter": 2000, "ftol": 1e-12}).x


def rates(theta, mats):
    Xk1, Xk2, Xs1, Xs2, Xd = mats
    pk, ps, pd_ = Xk1.shape[1], Xs1.shape[1], Xd.shape[1]
    bk, ak, bs, as_, bd, ad = M._unpack(theta, pk, ps, pd_)
    return (np.exp(np.clip(Xk1 @ bk + ak, -30, 6)),
            np.exp(np.clip(Xk2 @ bk + ak, -30, 6)),
            np.exp(np.clip(Xs1 @ bs + as_, -30, 6)),
            np.exp(np.clip(Xs2 @ bs + as_, -30, 6)),
            expit(np.clip(Xd @ bd + ad, -30, 30)))


def predict_v2(theta, mats, T, Cko, Csub, phik, phis):
    """
    Returns (P6, surv) where P6 is the 6-way distribution and surv[i, g] is
    P(fight i still live at the end of grid step g). The survival curve is
    what prices round totals and distance markets.
    """
    lk1, lk2, ls1, ls2, pa = rates(theta, mats)
    n = len(T)

    CK = Cko[1:]      # cumulative at each bin's right edge
    CS = Csub[1:]
    live = np.exp(-(np.multiply.outer(lk1 + lk2, CK) + np.multiply.outer(ls1 + ls2, CS)))
    prev = np.hstack([np.ones((n, 1)), live[:, :-1]])

    within = TGRID[None, :] < T[:, None]
    dko = prev * (phik[None, :] * (lk1 + lk2)[:, None]) * GRID * within
    dsb = prev * (phis[None, :] * (ls1 + ls2)[:, None]) * GRID * within

    p_ko = dko.sum(1)
    p_sub = dsb.sum(1)
    p_dec = np.exp(-(lk1 + lk2) * Phi_at(Cko, T) - (ls1 + ls2) * Phi_at(Csub, T))

    tot = p_ko + p_sub + p_dec
    p_ko, p_sub, p_dec = p_ko / tot, p_sub / tot, p_dec / tot

    fa_ko = lk1 / np.maximum(lk1 + lk2, 1e-12)
    fa_sb = ls1 / np.maximum(ls1 + ls2, 1e-12)

    P = np.column_stack([
        p_ko * fa_ko, p_sub * fa_sb, p_dec * pa,
        p_ko * (1 - fa_ko), p_sub * (1 - fa_sb), p_dec * (1 - pa),
    ])
    surv = np.where(within, live, 0.0)
    return P / P.sum(1, keepdims=True), surv


# ----------------------------------------------------------------------
# 3. Benchmark ladder
# ----------------------------------------------------------------------

def elo_table(fights, k=24.0, base=1500.0):
    """Point-in-time Elo. Returns per-fight pre-fight ratings."""
    r = {}
    out = []
    for f in fights.itertuples(index=False):
        ra = r.get(f.f1, base); rb = r.get(f.f2, base)
        out.append((f.EVENT, f.BOUT, f.f1, f.f2, ra, rb))
        ea = 1 / (1 + 10 ** ((rb - ra) / 400))
        sa = 1.0 if f.winner == f.f1 else 0.0
        r[f.f1] = ra + k * (sa - ea)
        r[f.f2] = rb + k * ((1 - sa) - (1 - ea))
    return pd.DataFrame(out, columns=["EVENT", "BOUT", "f1", "f2", "elo1", "elo2"])


def bench(kind, train, test, elo):
    n = len(test)
    if kind == "uniform":
        return np.full((n, 6), 1 / 6)

    if kind == "global":
        vc = train["method"].value_counts(normalize=True)
        row = np.array([vc.get("KO", .32) / 2, vc.get("SUB", .19) / 2,
                        (1 - vc.get("KO", .32) - vc.get("SUB", .19)) / 2] * 2)
        return np.tile(row / row.sum(), (n, 1))

    # division x round-format method rates
    P = np.zeros((n, 6))
    glob = train["method"].value_counts(normalize=True)
    for i, r in enumerate(test.itertuples(index=False)):
        sub = train[(train["division"] == r.division) &
                    ((train["max_min"] > 20) == (r.max_min > 20))]
        vc = sub["method"].value_counts(normalize=True) if len(sub) >= 40 else glob
        ko, sb = vc.get("KO", .32), vc.get("SUB", .19)
        dc = max(1 - ko - sb, 1e-6)
        P[i] = [ko, sb, dc, 0, 0, 0]

    if kind == "division":
        pa = np.full(n, 0.5)
    elif kind == "division_elo":
        pa = expit((test["elo_A"].values - test["elo_B"].values) / 173.7)
    else:
        raise ValueError(kind)

    out = np.column_stack([P[:, 0] * pa, P[:, 1] * pa, P[:, 2] * pa,
                           P[:, 0] * (1 - pa), P[:, 1] * (1 - pa), P[:, 2] * (1 - pa)])
    return out / out.sum(1, keepdims=True)


# ----------------------------------------------------------------------
# 4. Market collapses
# ----------------------------------------------------------------------

COLLAPSE = {
    "6way":      (lambda P: P, lambda y: np.array([CLASSES.index(v) for v in y])),
    "4way":      (lambda P: np.column_stack([P[:, 0] + P[:, 1], P[:, 2],
                                             P[:, 3] + P[:, 4], P[:, 5]]),
                  lambda y: np.array([{"A_KO": 0, "A_SUB": 0, "A_DEC": 1,
                                       "B_KO": 2, "B_SUB": 2, "B_DEC": 3}[v] for v in y])),
    "3way_method": (lambda P: np.column_stack([P[:, 0] + P[:, 3], P[:, 1] + P[:, 4],
                                               P[:, 2] + P[:, 5]]),
                    lambda y: np.array([["KO", "SUB", "DEC"].index(v.split("_")[1]) for v in y])),
    "distance":  (lambda P: np.column_stack([P[:, 2] + P[:, 5],
                                             P[:, 0] + P[:, 1] + P[:, 3] + P[:, 4]]),
                  lambda y: np.array([0 if v.endswith("DEC") else 1 for v in y])),
    "winner":    (lambda P: np.column_stack([P[:, :3].sum(1), P[:, 3:].sum(1)]),
                  lambda y: np.array([0 if v.startswith("A") else 1 for v in y])),
}


def ll_of(P, y, name):
    f, g = COLLAPSE[name]
    Q = f(P); Q = Q / Q.sum(1, keepdims=True)
    yi = g(y)
    return -np.log(np.clip(Q[np.arange(len(yi)), yi], 1e-15, 1))
