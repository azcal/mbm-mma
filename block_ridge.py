"""
Block-specific ridge. The single shared lambda forces a tradeoff: the
interaction terms need heavy shrinkage to help the method split, but the
main effects want almost none to keep the winner-side edge.
Penalise them separately.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
import mma_method_model as M
from mma_method_model import (
    load, build_fights, build_roundstats, build_dataset, randomize_corners,
    Scaler, predict, baseline, CLASSES, n_params,
)

rng = np.random.default_rng(0)

def _data(d="/tmp"):
    raw = load(d)
    ds = randomize_corners(build_dataset(build_fights(raw), build_roundstats(raw)))
    return ds, sorted(ds["division"].unique())

N_INT = 5  # interaction columns appended to each hazard block


def design_i(df, divs):
    Xk1, Xk2, Xs1, Xs2, Xd = M.design(df, divs)

    def ik(ap, bp):
        return np.column_stack([
            df[ap + "kd15"].values * df[bp + "lko"].values,
            df[ap + "kd15"].values * df[bp + "kda15"].values,
            df[ap + "wko"].values * df[bp + "lko"].values,
            df[ap + "slpm"].values * (1 - df[bp + "dfn"].values),
            df[ap + "acc"].values * df[bp + "kda15"].values,
        ])

    def isb(ap, bp):
        return np.column_stack([
            df[ap + "sub15"].values * df[bp + "lsub"].values,
            df[ap + "wsub"].values * df[bp + "lsub"].values,
            df[ap + "td15"].values * (1 - df[bp + "tddef"].values),
            df[ap + "ctrl"].values * df[bp + "lsub"].values,
            df[ap + "tdacc"].values * df[bp + "suba15"].values,
        ])

    return (np.hstack([Xk1, ik("A_", "B_")]), np.hstack([Xk2, ik("B_", "A_")]),
            np.hstack([Xs1, isb("A_", "B_")]), np.hstack([Xs2, isb("B_", "A_")]), Xd)


def nll_vec(theta, mats, t, T, y, pen):
    """Same likelihood, but pen is a per-parameter ridge weight vector."""
    Xk1, Xk2, Xs1, Xs2, Xd = mats
    pk, ps, pd_ = Xk1.shape[1], Xs1.shape[1], Xd.shape[1]
    bk, ak, bs, as_, bd, ad = M._unpack(theta, pk, ps, pd_)
    n = len(t)

    l_k1 = np.exp(np.clip(Xk1 @ bk + ak, -30, 6))
    l_k2 = np.exp(np.clip(Xk2 @ bk + ak, -30, 6))
    l_s1 = np.exp(np.clip(Xs1 @ bs + as_, -30, 6))
    l_s2 = np.exp(np.clip(Xs2 @ bs + as_, -30, 6))
    L = l_k1 + l_k2 + l_s1 + l_s2

    is_dec = (y == "A_DEC") | (y == "B_DEC")
    u = np.where(is_dec, T, t)
    i_k1 = (y == "A_KO").astype(float); i_k2 = (y == "B_KO").astype(float)
    i_s1 = (y == "A_SUB").astype(float); i_s2 = (y == "B_SUB").astype(float)

    ll = -L * u
    ll += i_k1 * np.log(l_k1 + 1e-300) + i_k2 * np.log(l_k2 + 1e-300)
    ll += i_s1 * np.log(l_s1 + 1e-300) + i_s2 * np.log(l_s2 + 1e-300)

    p_a = expit(np.clip(Xd @ bd + ad, -30, 30))
    a_dec = (y == "A_DEC"); b_dec = (y == "B_DEC")
    ll += a_dec * np.log(p_a + 1e-12) + b_dec * np.log(1 - p_a + 1e-12)

    f = -ll.sum() / n + float(pen @ (theta ** 2))

    w_k1 = i_k1 - u * l_k1; w_k2 = i_k2 - u * l_k2
    w_s1 = i_s1 - u * l_s1; w_s2 = i_s2 - u * l_s2
    gd = a_dec * (1 - p_a) - b_dec * p_a

    g = np.concatenate([
        -(Xk1.T @ w_k1 + Xk2.T @ w_k2) / n, [-(w_k1.sum() + w_k2.sum()) / n],
        -(Xs1.T @ w_s1 + Xs2.T @ w_s2) / n, [-(w_s1.sum() + w_s2.sum()) / n],
        -(Xd.T @ gd) / n, [-gd.sum() / n],
    ]) + 2 * pen * theta
    return f, g


def make_pen(mats, lam_main, lam_int):
    pk, ps, pd_ = mats[0].shape[1], mats[2].shape[1], mats[4].shape[1]
    pen = np.zeros(n_params(mats))
    pen[:pk] = lam_main
    pen[pk - N_INT:pk] = lam_int          # KO interactions
    pen[pk] = 0.0                          # intercept unpenalised
    s = pk + 1
    pen[s:s + ps] = lam_main
    pen[s + ps - N_INT:s + ps] = lam_int   # SUB interactions
    pen[s + ps] = 0.0
    d = s + ps + 1
    pen[d:d + pd_] = lam_main
    return pen


def fit_v(mats, t, T, y, pen):
    pk, ps = mats[0].shape[1], mats[2].shape[1]
    x0 = np.zeros(n_params(mats))
    x0[pk] = np.log(0.02); x0[pk + 1 + ps] = np.log(0.012)
    return minimize(nll_vec, x0, args=(mats, t, T, y, pen), jac=True,
                    method="L-BFGS-B", options={"maxiter": 2000, "ftol": 1e-12}).x


def walk(lam_main, lam_int, ds=None, divs=None):
    if ds is None:
        ds, divs = _data()
    P, B, Y = [], [], []
    for yr in sorted(y for y in ds["date"].dt.year.unique() if y >= 2015):
        tr = ds[ds["date"].dt.year < yr]; te = ds[ds["date"].dt.year == yr]
        if len(tr) < 800 or len(te) < 30:
            continue
        mtr, mte = design_i(tr, divs), design_i(te, divs)
        sc = [Scaler().fit(m) for m in mtr]
        mtr = tuple(s.t(m) for s, m in zip(sc, mtr))
        mte = tuple(s.t(m) for s, m in zip(sc, mte))
        pen = make_pen(mtr, lam_main, lam_int)
        th = fit_v(mtr, tr["end_min"].values, tr["max_min"].values, tr["target"].values, pen)
        P.append(predict(th, mte, te["max_min"].values))
        B.append(baseline(tr, te))
        Y.append(te["target"].values)
    return np.vstack(P), np.vstack(B), np.concatenate(Y)


def meth_loss(P, y):
    Pm = np.column_stack([P[:, [0, 3]].sum(1), P[:, [1, 4]].sum(1), P[:, [2, 5]].sum(1)])
    Pm /= Pm.sum(1, keepdims=True)
    o = ["KO", "SUB", "DEC"]
    ym = np.array([o.index(v.split("_")[1]) for v in y])
    return -np.log(np.clip(Pm[np.arange(len(ym)), ym], 1e-15, 1))


def full_loss(P, y):
    yi = np.array([CLASSES.index(v) for v in y])
    return -np.log(np.clip(P[np.arange(len(yi)), yi], 1e-15, 1))


def boot(d, n=6000):
    b = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return np.percentile(b, 2.5), np.percentile(b, 97.5), (b >= 0).mean()


if __name__ == "__main__":
    ds, divs = _data()
    print(f"{'lam_main':>9} {'lam_int':>8} | {'6way_ll':>8} {'vs_base':>8} | {'method_d':>9} {'winner_d':>9}")
    print("-" * 62)
    results = []
    for lm in [0.0, 0.002, 0.01]:
        for li in [0.0, 0.02, 0.08, 0.3, 1.0]:
            P, B, y = walk(lm, li)
            fl = full_loss(P, y).mean()
            bl = full_loss(B, y).mean()
            md = (meth_loss(P, y) - meth_loss(B, y)).mean()
            Pw = np.column_stack([P[:, :3].sum(1), P[:, 3:].sum(1)])
            yw = np.array([0 if v.startswith("A") else 1 for v in y])
            wd = (-np.log(np.clip(Pw[np.arange(len(yw)), yw], 1e-15, 1))).mean() - np.log(2)
            print(f"{lm:>9} {li:>8} | {fl:>8.4f} {fl - bl:>+8.4f} | {md:>+9.4f} {wd:>+9.4f}")
            results.append((lm, li, fl - bl, md, P, y, B))

    best = min(results, key=lambda r: r[2])
    print(f"\nBest overall: lam_main={best[0]} lam_int={best[1]}  6-way delta {best[2]:+.4f}")
    P, y, B = best[4], best[5], best[6]
    d_full = full_loss(P, y) - full_loss(B, y)
    lo, hi, p = boot(d_full)
    print(f"  6-way delta   95% CI [{lo:+.4f}, {hi:+.4f}]  P(>=0)={p:.4f}")
    d_m = meth_loss(P, y) - meth_loss(B, y)
    lo, hi, p = boot(d_m)
    print(f"  method delta  {d_m.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  P(>=0)={p:.4f}")

    best_m = min(results, key=lambda r: r[3])
    print(f"\nBest method-side: lam_main={best_m[0]} lam_int={best_m[1]}  method delta {best_m[3]:+.4f}")
    d_m2 = meth_loss(best_m[4], best_m[5]) - meth_loss(best_m[6], best_m[5])
    lo, hi, p = boot(d_m2)
    print(f"  95% CI [{lo:+.4f}, {hi:+.4f}]  P(>=0)={p:.4f}")
