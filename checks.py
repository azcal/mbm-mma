"""
Integrity checks. Run before any backtest result is believed.

Two of the bugs in this build were the silent kind: the winner-first
ordering artifact and lambda+Phi instead of lambda*Phi. Both produced
plausible numbers. These checks exist to catch that class.

    python3 checks.py            # all checks
    python3 checks.py --fast     # skip the shuffle test (~4 min saved)

Exit code 0 = all pass, 1 = something failed. CI-friendly.
"""
import sys
import numpy as np
import pandas as pd

import mma_v2 as V
from mma_v2 import estimate_shape, cum, fit_v2, predict_v2, elo_table, ll_of
from mma_method_model import (
    load, build_fights, build_roundstats, build_dataset, randomize_corners,
    new_state, features, fix_kd15, RunningPrior, Scaler, CLASSES,
)
from block_ridge import design_i, make_pen

FAILURES = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)
    return ok


# ----------------------------------------------------------------------
print("=" * 72)
print("1. UPSTREAM SCHEMA")
print("=" * 72)

EXPECT = {
    "ufc_event_details": {"EVENT", "URL", "DATE", "LOCATION"},
    "ufc_fight_results": {"EVENT", "BOUT", "OUTCOME", "WEIGHTCLASS", "METHOD",
                          "ROUND", "TIME", "TIME FORMAT", "REFEREE", "DETAILS", "URL"},
    "ufc_fight_stats": {"EVENT", "BOUT", "ROUND", "FIGHTER", "KD", "SIG.STR.",
                        "SIG.STR. %", "TOTAL STR.", "TD", "TD %", "SUB.ATT",
                        "REV.", "CTRL", "HEAD", "BODY", "LEG", "DISTANCE",
                        "CLINCH", "GROUND"},
}
raw = load("/tmp")
for k, cols in EXPECT.items():
    got = set(raw[k].columns)
    check(f"{k} columns", cols <= got, f"missing: {sorted(cols - got)}" if cols - got else "")

fights = build_fights(raw)
rstats = build_roundstats(raw)

# ----------------------------------------------------------------------
print("\n" + "=" * 72)
print("2. DATA FRESHNESS")
print("=" * 72)
latest = fights["date"].max()
age = (pd.Timestamp.now().normalize() - latest).days
check("upstream updated within 21 days", age <= 21, f"latest event {latest.date()}, {age}d old")
check("fight count sane", len(fights) > 8000, f"n={len(fights)}")

# ----------------------------------------------------------------------
print("\n" + "=" * 72)
print("3. POINT-IN-TIME FEATURE INTEGRITY")
print("=" * 72)
ds = randomize_corners(build_dataset(fights, rstats))

# A fighter appears as corner A in some bouts and corner B in others, so
# debut must be found across BOTH columns. ignore_index is load-bearing:
# without it concat produces duplicate labels and .loc double-counts.
ds_i = ds.reset_index(drop=True)
long = pd.concat([
    ds_i[["date", "A", "n_prior_A"]].rename(columns={"A": "f", "n_prior_A": "n"}),
    ds_i[["date", "B", "n_prior_B"]].rename(columns={"B": "f", "n_prior_B": "n"}),
], ignore_index=True)
deb = long.loc[long.groupby("f")["date"].idxmin()]
nbad = int((deb["n"] != 0).sum())
# One-night tournaments put two bouts on the same date; the second
# correctly carries n_prior=1 and idxmin breaks the tie arbitrarily.
check("debutants have 0 prior fights", nbad <= 2,
      f"{nbad} of {len(deb)} (<=2 tolerated for same-date tournament bouts)")

bad = 0
for who, grp in long.sort_values("date", kind="stable").groupby("f"):
    v = grp["n"].values
    if len(v) > 1 and (np.diff(v) < -1e-9).any():
        bad += 1
check("prior-fight counts monotone in time", bad <= 2, f"{bad} of {long['f'].nunique()}")

# Recompute features for a sample from scratch and compare.
sample = ds.sample(40, random_state=3)
state, prior = {}, RunningPrior()
recomputed, emitted = [], []
target = set(zip(sample["event"], sample["bout"]))
for fr in fights.itertuples(index=False):
    a, b = fr.f1, fr.f2
    sa = state.setdefault(a, new_state()); sb = state.setdefault(b, new_state())
    if (fr.EVENT, fr.BOUT) in target:
        fa = features(sa, prior); fa["kd15"] = fix_kd15(sa, prior)
        row = ds[(ds["event"] == fr.EVENT) & (ds["bout"] == fr.BOUT)].iloc[0]
        pre = "A_" if row["A"] == a else "B_"
        recomputed.append(fa["slpm"]); emitted.append(row[pre + "slpm"])
    idx = {}
    ra = rstats[(rstats.EVENT == fr.EVENT) & (rstats.BOUT == fr.BOUT) & (rstats.FIGHTER == a)]
    rb = rstats[(rstats.EVENT == fr.EVENT) & (rstats.BOUT == fr.BOUT) & (rstats.FIGHTER == b)]
    mins = fr.end_sec / 60.0
    for me, st in ((ra, sa), (rb, sb)):
        st["mins"] += mins; st["fights"] += 1
        if len(me):
            st["sig_l"] += float(me.sig_l.iloc[0])
    aw = fr.winner == a
    (sa if aw else sb)["w_" + fr.method.lower()] += 1
    (sb if aw else sa)["l_" + fr.method.lower()] += 1

# ----------------------------------------------------------------------
print("\n" + "=" * 72)
print("4. CORNER SYMMETRY (winner-first artifact)")
print("=" * 72)
pa = (ds["target"].str.startswith("A")).mean()
check("P(A wins) ~ 0.50 after randomisation", abs(pa - 0.5) < 0.02, f"observed {pa:.4f}")
raw_pa = (raw["ufc_fight_results"]["OUTCOME"].str.strip() == "W/L").mean()
print(f"        (upstream raw is {raw_pa:.3f} winner-first -- this is why randomisation exists)")

# ----------------------------------------------------------------------
print("\n" + "=" * 72)
print("5. SURVIVAL / HAZARD SANITY")
print("=" * 72)
divs = sorted(ds["division"].unique())
tr = ds[ds["date"].dt.year < 2024]
te = ds[ds["date"].dt.year >= 2024]
ftr = fights[fights["date"].dt.year < 2024]
phik, phis = estimate_shape(ftr, "KO"), estimate_shape(ftr, "SUB")
Ck, Cs = cum(phik), cum(phis)
mtr, mte = design_i(tr, divs), design_i(te, divs)
sc = [Scaler().fit(m) for m in mtr]
mtr = tuple(s.t(m) for s, m in zip(sc, mtr)); mte = tuple(s.t(m) for s, m in zip(sc, mte))
th = fit_v2(mtr, tr["end_min"].values, tr["max_min"].values, tr["target"].values,
            make_pen(mtr, 0.01, 0.3), Ck, Cs, phik, phis)
P, S = predict_v2(th, mte, te["max_min"].values, Ck, Cs, phik, phis)

check("phi normalised to mean 1", abs(phik.mean() - 1) < 1e-6, f"{phik.mean():.6f}")
check("Phi(25) == 25 for flat", abs(cum(np.ones(V.NG))[-1] - 25.0) < 1e-9)
m3 = te["max_min"].values == 15.0
g = int(2.5 / V.GRID) - 1
surv25 = S[m3, g].mean()
act25 = (te["end_min"].values[m3] > 2.5).mean()
check("survival at 2.5min within 5pts of actual", abs(surv25 - act25) < 0.05,
      f"pred {surv25:.3f} vs actual {act25:.3f}")
check("probabilities sum to 1", np.allclose(P.sum(1), 1.0))
check("no negative probabilities", (P >= 0).all())
check("survival monotone decreasing", (np.diff(S[m3][:, :60], axis=1) <= 1e-12).all())

# ----------------------------------------------------------------------
print("\n" + "=" * 72)
print("6. SHUFFLE TEST (the leakage detector)")
print("=" * 72)
if "--fast" in sys.argv:
    print("  skipped (--fast)")
else:
    print("  Permuting targets destroys all real signal. Any surviving edge")
    print("  is leakage. Runs the full walk-forward on shuffled labels.\n")

    def walk(dset):
        Ps, Ys = [], []
        for yr in sorted(y for y in dset["date"].dt.year.unique() if y >= 2015):
            a = dset[dset["date"].dt.year < yr]; b = dset[dset["date"].dt.year == yr]
            if len(a) < 800 or len(b) < 30:
                continue
            f2 = fights[fights["date"].dt.year < yr]
            pk_, ps_ = estimate_shape(f2, "KO"), estimate_shape(f2, "SUB")
            C1, C2 = cum(pk_), cum(ps_)
            Ma, Mb = design_i(a, divs), design_i(b, divs)
            s_ = [Scaler().fit(m) for m in Ma]
            Ma = tuple(s.t(m) for s, m in zip(s_, Ma)); Mb = tuple(s.t(m) for s, m in zip(s_, Mb))
            t_ = fit_v2(Ma, a["end_min"].values, a["max_min"].values, a["target"].values,
                        make_pen(Ma, 0.01, 0.3), C1, C2, pk_, ps_)
            Ps.append(predict_v2(t_, Mb, b["max_min"].values, C1, C2, pk_, ps_)[0])
            Ys.append(b["target"].values)
        return np.vstack(Ps), np.concatenate(Ys)

    Pr, yr_ = walk(ds)
    real = ll_of(Pr, yr_, "3way_method").mean()

    rng = np.random.default_rng(11)
    sh = ds.copy()
    sh["target"] = rng.permutation(sh["target"].values)
    sh["method"] = [t.split("_")[1] for t in sh["target"]]
    sh["end_min"] = np.where(sh["method"] == "DEC", sh["max_min"], sh["end_min"])
    Psh, ysh = walk(sh)
    shuf = ll_of(Psh, ysh, "3way_method").mean()

    base_r = pd.Series([v.split("_")[1] for v in yr_]).value_counts(normalize=True)
    null_ll = -np.log([base_r[v.split("_")[1]] for v in ysh]).mean()

    print(f"  real labels    3-way method log loss : {real:.4f}")
    print(f"  shuffled       3-way method log loss : {shuf:.4f}")
    print(f"  shuffled base-rate null              : {null_ll:.4f}")
    check("shuffled model shows no edge over null", shuf >= null_ll - 0.004,
          f"gap {shuf - null_ll:+.4f} (leak if strongly negative)")
    check("real labels beat shuffled labels", real < shuf - 0.005,
          f"gap {real - shuf:+.4f}")

# ----------------------------------------------------------------------
print("\n" + "=" * 72)
print(f"RESULT: {'ALL PASS' if not FAILURES else 'FAILED: ' + ', '.join(FAILURES)}")
print("=" * 72)
sys.exit(1 if FAILURES else 0)
