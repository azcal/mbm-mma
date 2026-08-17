"""
Is the method market silent because the patch is not running, or because
nothing clears the bar?

The Discord embed cannot answer that: it only ever shows a METH line when a
pick qualifies. This reads your existing log, which already carries
p_a_ko / p_a_sub / p_a_dec / p_b_ko / p_b_sub / p_b_dec, and computes the
lift distribution directly. No pipeline run needed.

    python diagnose_method.py logs/mma_log.csv
    python diagnose_method.py logs/graded.csv
    python diagnose_method.py logs/            # scans the directory

Tells you:
  - whether the six method columns exist at all
  - the lift distribution across every logged bout
  - how many bouts clear 1.35 / 1.20 / 1.10
  - what METH_LIFT_MIN would produce a given number of picks
  - whether the method head is calibrated, on rows that have results
"""
import glob
import os
import sys
import numpy as np
import pandas as pd

CLS = ("A_KO", "A_SUB", "A_DEC", "B_KO", "B_SUB", "B_DEC")
PCOLS = ["p_a_ko", "p_a_sub", "p_a_dec", "p_b_ko", "p_b_sub", "p_b_dec"]
LO, HI = 0.08, 0.26


def find_logs(path):
    if os.path.isdir(path):
        return sorted(glob.glob(os.path.join(path, "*.csv")))
    return [path]


def load(paths):
    best, bestn = None, -1
    for p in paths:
        try:
            d = pd.read_csv(p)
        except Exception:
            continue
        have = sum(c in d.columns for c in PCOLS)
        if have == 6 and len(d) > bestn:
            best, bestn = (d, p), len(d)
    return best


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    hit = load(find_logs(sys.argv[1]))
    if hit is None:
        print("No CSV found carrying all six method columns:")
        print("  " + ", ".join(PCOLS))
        print("\nThat alone is the answer: the engine's method probabilities are not")
        print("reaching the log, so the METH line can never fire. Check that")
        print("E.predict() returns them and that the log merge is not dropping them")
        print("(see the 'log columns seeded' edit in patch_daily_mma.py).")
        for p in find_logs(sys.argv[1]):
            try:
                cols = list(pd.read_csv(p, nrows=1).columns)
            except Exception:
                continue
            print(f"\n  {os.path.basename(p)}: {len(cols)} cols")
            print(f"    method cols present: {[c for c in PCOLS if c in cols] or 'none'}")
        return 1

    d, src = hit
    print(f"reading {src}   {len(d)} row(s)\n")

    P = d[PCOLS].to_numpy(dtype=float)
    ok = np.isfinite(P).all(1) & (P.sum(1) > 0)
    d, P = d[ok].reset_index(drop=True), P[ok]
    P = P / P.sum(1, keepdims=True)
    print(f"{len(P)} row(s) with usable method probabilities")

    print("\nmethod probability spread (are they shrunk flat?)")
    print(f"  {'class':<7} {'mean':>7} {'sd':>7} {'min':>7} {'max':>7}")
    for j, c in enumerate(CLS):
        print(f"  {c:<7} {P[:, j].mean():>7.3f} {P[:, j].std():>7.3f} "
              f"{P[:, j].min():>7.3f} {P[:, j].max():>7.3f}")

    # base rates: division x rounds if available, else global
    if "p_win_a" not in d.columns:
        print("\nno p_win_a column, cannot build the winner-adjusted base. stopping.")
        return 1
    pa = d.p_win_a.to_numpy(dtype=float).clip(0.02, 0.98)

    ko = P[:, [0, 3]].sum(1).mean()
    sb = P[:, [1, 4]].sum(1).mean()
    dc = max(1 - ko - sb, 1e-6)
    glob = np.array([ko, sb, dc])
    tbl = {}
    if {"wt", "sched"} <= set(d.columns):
        for (w, s), g in d.groupby(["wt", "sched"]):
            if len(g) >= 40:
                i = g.index.to_numpy()
                tbl[(w, s)] = np.array([P[i][:, [0, 3]].sum(1).mean(),
                                        P[i][:, [1, 4]].sum(1).mean(), 0.0])
                tbl[(w, s)][2] = max(1 - tbl[(w, s)][:2].sum(), 1e-6)
        print(f"\nbase-rate table built for {len(tbl)} division x rounds cell(s)")
    else:
        print("\nno wt/sched columns, using a global base rate")

    B = np.zeros_like(P)
    for i in range(len(P)):
        v = tbl.get((d.wt.iloc[i], d.sched.iloc[i]), glob) if tbl else glob
        B[i] = [v[0] * pa[i], v[1] * pa[i], v[2] * pa[i],
                v[0] * (1 - pa[i]), v[1] * (1 - pa[i]), v[2] * (1 - pa[i])]
    B = B / B.sum(1, keepdims=True)

    lift = np.where((P >= LO) & (P <= HI), P / np.maximum(B, 1e-9), -1.0)
    best = lift.max(1)
    inband = (lift > 0).any(1)

    print(f"\n{inband.sum()} of {len(P)} bout(s) have at least one outcome inside "
          f"the {LO}-{HI} band")
    print("\nbest-lift distribution across all bouts")
    for q in [50, 75, 90, 95, 99, 100]:
        print(f"  p{q:<3} {np.percentile(best[best > 0], q) if inband.sum() else 0:.3f}x")

    print("\npicks that would post, by threshold")
    print(f"  {'thresh':>7} {'picks':>7} {'rate':>7}")
    for t in [1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35, 1.45, 1.60]:
        n = int((best >= t).sum())
        print(f"  {t:>7.2f} {n:>7} {n / len(P):>6.1%}")

    if (best >= 1.35).sum() == 0:
        print("\n>>> NOTHING clears 1.35x. The patch can be perfectly applied and the")
        print(">>> Discord board will still be byte-identical. Your engine's method")
        print(">>> head is too compressed toward its own base rate to separate.")
        for t in [1.30, 1.20, 1.15, 1.10, 1.05]:
            if (best >= t).sum() >= max(3, int(0.05 * len(P))):
                print(f">>> Lower METH_LIFT_MIN to {t:.2f} to post "
                      f"{int((best >= t).sum())} ({(best >= t).mean():.0%}).")
                break
        else:
            print(">>> No threshold gives useful volume. Use mbm_mma_method.py instead;")
            print(">>> its measured lift is 1.86x at 1.35x on the same selection rule.")

    # calibration, if results are present
    if "result" in d.columns and "m" in d.columns:
        g = d[d.result.isin([0.0, 1.0]) & d.m.notna()].copy()
        if len(g) >= 30:
            mm = g.m.replace({"TKO": "KO"})
            idx = g.index.to_numpy()
            ai = np.array([CLS.index(("A_" if r == 1.0 else "B_") + m)
                           if ("A_" if r == 1.0 else "B_") + m in CLS else -1
                           for r, m in zip(g.result.to_numpy(), mm.to_numpy())])
            keep = ai >= 0
            if keep.sum() >= 30:
                Pg, Bg, aig = P[idx][keep], B[idx][keep], ai[keep]
                n = len(aig)
                mll = -np.log(np.clip(Pg[np.arange(n), aig], 1e-12, 1)).mean()
                bll = -np.log(np.clip(Bg[np.arange(n), aig], 1e-12, 1)).mean()
                print(f"\nMETHOD HEAD, GRADED ON {n} BOUT(S)  (never measured before)")
                print(f"  6-way log loss   model {mll:.4f}   base {bll:.4f}   "
                      f"delta {mll - bll:+.4f}")
                lg = np.where((Pg >= LO) & (Pg <= HI),
                              Pg / np.maximum(Bg, 1e-9), -1.0)
                sel = lg.argmax(1); bl = lg.max(1)
                for t in [1.20, 1.35]:
                    m = bl >= t
                    if m.sum() >= 10:
                        hits = (sel[m] == aig[m]).mean()
                        pred = Pg[m, sel[m]].mean()
                        base = Bg[m, sel[m]].mean()
                        print(f"  at {t:.2f}x: n={m.sum():3d}  pred {pred:.3f}  "
                              f"actual {hits:.3f}  gap {hits - pred:+.3f}  "
                              f"lift {hits / max(base, 1e-9):.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
