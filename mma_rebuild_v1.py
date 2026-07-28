"""
MBM MMA rebuild v1. Tune on 2018-2025 walk-forward, freeze, open 2026 exactly once.
"""
import json, sys, numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, "/home/claude")
import engine_mma as E

F = pd.read_csv("/home/claude/mma/fights_clean.csv")
A = E.load_attrs("/home/claude/mma/attrs.csv")
P = E.prepare(F, A)


def ll(y, p):
    p = np.clip(np.asarray(p, float), 1e-6, 1-1e-6); y = np.asarray(y, float)
    return -(y*np.log(p) + (1-y)*np.log(1-p))


def dm(y, a, b):
    v = ll(y, b) - ll(y, a); se = v.std(ddof=1)/np.sqrt(len(v)); t = v.mean()/se
    return v.mean(), t, 1 - stats.t.cdf(t, len(v)-1)


def walk(D, C, years):
    """Walk-forward: train on everything before the year, predict UFC fights in it."""
    out = []
    for yr in years:
        tr = D.date.dt.year < yr
        te = (D.promotion == "UFC") & (D.date.dt.year == yr)
        if te.sum() < 50:
            continue
        heads = E.fit_heads(D, tr, C)
        pr = E.predict(heads, D[te])
        out.append(pd.DataFrame({
            "y": D.loc[te, "y"].values.astype(float),
            "dist": (D.loc[te, "meth"] == "DEC").astype(float).values,
            "p_win": pr.p_win_a.values, "p_dist": pr.p_distance.values,
            "elo": (1/(1+10**(-D.loc[te, "elo_d"].values/400)))}))
    return pd.concat(out, ignore_index=True)


DEV = range(2018, 2026)
print("TUNING on the 2018-2025 development window")
grid = []
for K in (16, 24, 32):
    for hl in (4, 6, 9):
        for fin in (1.0, 1.25):
            cfg = {"K": K, "hl": hl, "tau": 2.0, "finish_bonus": fin}
            D = E.build(P, A, cfg)
            for C in (0.05, 0.2, 1.0):
                r = walk(D, C, DEV)
                grid.append((ll(r.y, r.p_win).mean(), K, hl, fin, C))
grid.sort()
print(f"  {'LL':>9} {'K':>3} {'hl':>3} {'finish':>7} {'C':>5}")
for v, K, hl, fin, C in grid[:5]:
    print(f"  {v:>9.5f} {K:>3} {hl:>3} {fin:>7} {C:>5}")
print(f"  configs searched: {len(grid)}   spread: {grid[-1][0]-grid[0][0]:.5f}")

_, K, hl, fin, C = grid[0]
CFG = {"K": K, "hl": hl, "tau": 2.0, "finish_bonus": fin}
SPEC = {"cfg": CFG, "C": C, "train_through": 2025,
        "note": "tuned on 2018-2025 walk-forward; 2026 opened once"}
json.dump(SPEC, open("/home/claude/mma/spec.json", "w"), indent=1)
print(f"\nFROZEN SPEC: {SPEC}")

D = E.build(P, A, CFG)
dev = walk(D, C, DEV)
print(f"\nDEV RESULT (2018-2025, n={len(dev)}):")
for nm, col, tgt in (("moneyline", "p_win", "y"), ("distance", "p_dist", "dist")):
    y = dev[tgt].values; p = dev[col].values
    pb = np.full(len(y), dev[tgt].mean() if nm == "distance" else 0.5)
    e, t, _ = dm(y, p, pb)
    print(f"  {nm:10s} LL={ll(y,p).mean():.5f}  base={ll(y,pb).mean():.5f}  "
          f"edge={e:+.5f}  t={t:+.2f}  skill={1-ll(y,p).mean()/ll(y,pb).mean():.2%}")

print("\n" + "="*74)
print("OPENING 2026. This is the single read.")
tr = D.date.dt.year <= 2025
te = (D.promotion == "UFC") & (D.date.dt.year == 2026)
heads = E.fit_heads(D, tr, C)
pr = E.predict(heads, D[te])
y = D.loc[te, "y"].values.astype(float)
dist = (D.loc[te, "meth"] == "DEC").astype(float).values
elo = 1/(1+10**(-D.loc[te, "elo_d"].values/400))
print(f"  n={len(y)}   A-wins {y.mean():.4f}   distance rate {dist.mean():.4f}")
print(f"\n  {'market':12s} {'model':>9} {'floor':>9} {'base':>9} {'edge':>9} {'t':>6} {'skill':>7}")
pb = np.full(len(y), 0.5)
e, t, _ = dm(y, elo, pb)
print(f"  {'ML: Elo':12s} {ll(y,elo).mean():>9.5f} {'':>9} {ll(y,pb).mean():>9.5f} {e:>+9.5f} {t:>+6.2f}")
e, t, _ = dm(y, pr.p_win_a.values, pb)
print(f"  {'ML: model':12s} {ll(y,pr.p_win_a).mean():>9.5f} {ll(y,elo).mean():>9.5f} "
      f"{ll(y,pb).mean():>9.5f} {e:>+9.5f} {t:>+6.2f} "
      f"{1-ll(y,pr.p_win_a).mean()/ll(y,pb).mean():>6.2%}")
e, t, _ = dm(y, pr.p_win_a.values, elo)
print(f"    layer over Elo floor: {e:+.5f}  t={t:+.2f}")
db = np.full(len(dist), D.loc[tr & (D.promotion == 'UFC'), 'meth'].eq('DEC').mean())
e, t, _ = dm(dist, pr.p_distance.values, db)
print(f"  {'distance':12s} {ll(dist,pr.p_distance).mean():>9.5f} {'':>9} {ll(dist,db).mean():>9.5f} "
      f"{e:>+9.5f} {t:>+6.2f} {1-ll(dist,pr.p_distance).mean()/ll(dist,db).mean():>6.2%}")

# method: 3-way log loss against the training-set method distribution
cls = ["KO", "SUB", "DEC"]
act = D.loc[te, "meth"].values
pm = np.c_[pr.p_a_ko + pr.p_b_ko, pr.p_a_sub + pr.p_b_sub, pr.p_a_dec + pr.p_b_dec]
pm = pm / pm.sum(axis=1, keepdims=True)
prior = np.array([(D.loc[tr, "meth"] == c).mean() for c in cls])
idx = np.array([cls.index(m) for m in act])
mll = -np.log(np.clip(pm[np.arange(len(idx)), idx], 1e-6, 1)).mean()
bll = -np.log(prior[idx]).mean()
print(f"  {'method (3way)':12s} {mll:>9.5f} {'':>9} {bll:>9.5f} {bll-mll:>+9.5f} "
      f"{'':>6} {1-mll/bll:>6.2%}")

out = D.loc[te, ["date", "event", "promotion", "A", "B", "wt", "sched", "y", "meth"]].copy()
out = out.join(pr)
out.to_csv("/home/claude/mma/eval_2026.csv", index=False)
print(f"\nwritten eval_2026.csv ({len(out)} rows)")
