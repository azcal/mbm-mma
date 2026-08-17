"""
Patch daily_mma.py to surface and grade the method market.

    python patch_daily_mma.py path/to/daily_mma.py

Idempotent. Writes daily_mma.py.bak first. Run with --revert to undo.

WHAT THIS FIXES

daily_mma.py already computes the full 6-way method distribution:
p_a_ko, p_a_sub, p_a_dec, p_b_ko, p_b_sub, p_b_dec come straight out of
E.predict(). The docstring already promises "three markets: moneyline,
goes-the-distance, and winner by method".

The third market is never printed. BF.mma_board() collapses those six
numbers into the two-row form (winner + distance) and the method split is
discarded. It is also never graded: the rolling block reports moneyline
and distance only, so the method head has been running ungraded the whole
time.

This patch:
  1. adds a METH line using the backtested selection rule
  2. logs meth_pick / meth_p / meth_base so it can be graded
  3. adds method to the rolling grade block
  4. appends the method line to the Discord combo

It uses YOUR engine's method probabilities, not the hazard model. Grade
it for a season first. If it underperforms, mbm_mma_method.py is the
drop-in replacement and INTEGRATION.md covers the swap.
"""
import os
import re
import shutil
import sys

MARK = "# --- MBM method market (patched) ---"

HELPERS = '''
# --- MBM method market (patched) ---
# The engine already returns a full 6-way method distribution; it was being
# collapsed to winner+distance by BF.mma_board and never shown or graded.
#
# Selection rule is NOT argmax. Argmax picks whichever outcome has the highest
# base rate, which is decisions, and a decision pick is what you would have
# guessed with no model at all (measured lift 1.03x over 4,695 fights). Instead
# post the outcome that departs most from the division base rate, inside the
# calibrated band, and only if the departure clears METH_LIFT_MIN. Measured
# 1.86x lift at 1.35x, posting on 9-14% of bouts. See BOARD.md.
METH_LO, METH_HI = 0.08, 0.26      # calibrated trust band, see CALIBRATION.md
METH_LIFT_MIN = 1.35               # volume/signal knob; 1.20 -> ~44% posting at 1.53x
METH_CLASSES = ("A_KO", "A_SUB", "A_DEC", "B_KO", "B_SUB", "B_DEC")


def _meth_base_rates(Dall, tr):
    """Division x scheduled-rounds method base rates from the training slice."""
    d = Dall.loc[tr]
    d = d[d.m.isin(["KO", "TKO", "SUB", "DEC"])] if "m" in d.columns else d
    out = {}
    if "m" not in d.columns:
        return out, (0.32, 0.19, 0.49)
    mm = d.m.replace({"TKO": "KO"})
    g = mm.value_counts(normalize=True)
    glob = (float(g.get("KO", .32)), float(g.get("SUB", .19)),
            max(1.0 - float(g.get("KO", .32)) - float(g.get("SUB", .19)), 1e-6))
    if "wt" in d.columns and "sched" in d.columns:
        for (wt, sc), grp in d.groupby(["wt", "sched"]):
            if len(grp) >= 40:
                v = grp.m.replace({"TKO": "KO"}).value_counts(normalize=True)
                ko, sb = float(v.get("KO", .32)), float(v.get("SUB", .19))
                out[(wt, int(sc))] = (ko, sb, max(1.0 - ko - sb, 1e-6))
    return out, glob


def _meth_pick(r, base_tbl, glob, lift_min=METH_LIFT_MIN):
    """Return (label, p, base, lift) or None when nothing clears the bar."""
    P = np.array([r.p_a_ko, r.p_a_sub, r.p_a_dec,
                  r.p_b_ko, r.p_b_sub, r.p_b_dec], dtype=float)
    s = P.sum()
    if not np.isfinite(s) or s <= 0:
        return None
    P = P / s
    pa = float(r.p_win_a)
    ko, sb, dc = base_tbl.get((r.wt, int(r.sched)), glob)
    B = np.array([ko * pa, sb * pa, dc * pa,
                  ko * (1 - pa), sb * (1 - pa), dc * (1 - pa)], dtype=float)
    B = B / B.sum()
    lift = np.where((P >= METH_LO) & (P <= METH_HI), P / np.maximum(B, 1e-9), -1.0)
    j = int(lift.argmax())
    if lift[j] < lift_min:
        return None
    cls = METH_CLASSES[j]
    who = r.A if cls.startswith("A_") else r.B
    return (f"{who} by {cls.split('_')[1]}", float(P[j]), float(B[j]), float(lift[j]))
# --- end MBM method market ---

'''

BOARD_CALL = '''            meth = _meth_pick(r, _meth_base, _meth_glob)
            if meth:
                _mlbl, _mp, _mb, _mlift = meth
                print(f"  METH {_mlbl} {_mp:.3f} -> needs {min_odds(_mp):.2f}"
                      f"  (base {_mb:.3f}, {_mlift:.2f}x)")
'''

COMBO_APPEND = '''            if meth and not close_display and n_min >= ADEQ_MIN_BOUTS:
                combo = list(combo) + [
                    f"{_mlbl}  {_mp*100:.1f}%  find {1/_mp:.2f} / "
                    f"{BF.american(_mp)} or better"]
'''

GRADE = '''        # method market: graded only on rows where a pick was actually posted
        if "meth_pick" in dec.columns:
            dm = dec[dec.meth_pick.notna() & dec.meth_hit.notna()]
            if len(dm) >= 30:
                ym = dm.meth_hit.astype(int).values
                pm = np.clip(dm.meth_p.values.astype(float), 1e-6, 1-1e-6)
                bm = np.clip(dm.meth_base.values.astype(float), 1e-6, 1-1e-6)
                print(f"  method   : LL={log_loss(ym, pm):.4f}  "
                      f"base={log_loss(ym, bm):.4f}  "
                      f"pred={pm.mean():.3f}  hit={ym.mean():.3f}  "
                      f"lift={ym.mean()/max(bm.mean(),1e-9):.2f}x  n={len(ym)}")
                if ym.mean() + 0.02 < pm.mean():
                    alerts.append("MMA method (running hot)")
'''


def patch(path):
    src = open(path, encoding="utf-8").read()
    if MARK in src:
        print("already patched, nothing to do")
        return 0
    shutil.copy(path, path + ".bak")
    out = src

    # 1. helpers, before def preflight()
    out = out.replace("\ndef preflight():", HELPERS + "\ndef preflight():", 1)

    # 2. base-rate table, right after heads are fit
    out = out.replace(
        "    heads = E.fit_heads(Dall, tr, C)",
        "    heads = E.fit_heads(Dall, tr, C)\n"
        "    _meth_base, _meth_glob = _meth_base_rates(Dall, tr)", 1)

    # 3. METH console line, after the DIST print
    anchor = ('        print(f"  DIST goes the distance {r.p_distance:.3f} -> needs '
              '{min_odds(r.p_distance):.2f}"\n'
              '                  f"  | inside {1-r.p_distance:.3f} -> needs '
              '{min_odds(1-r.p_distance):.2f}")\n')
    anchor = anchor.replace("        print", "            print").replace(
        "                  f", "                  f")
    m = re.search(r"[ ]+print\(f\"  DIST goes the distance.*?\n.*?\n", out, re.S)
    if not m:
        print("FAIL: could not locate the DIST print block"); return 1
    out = out[:m.end()] + BOARD_CALL + out[m.end():]

    # 4. method line appended to the Discord combo
    m2 = re.search(r"[ ]+combo = \[\"  too close to call\"\] \+ dist_lines\n", out)
    if not m2:
        print("FAIL: could not locate the combo assembly"); return 1
    out = out[:m2.end()] + COMBO_APPEND + out[m2.end():]

    # 5. persist method fields so they can be graded next run
    m3 = re.search(r"([ ]+)rows\.append\(\{\"date\": r\.date, \"game\":", out)
    if not m3:
        print("FAIL: could not locate rows.append"); return 1
    ind = m3.group(1)
    out = out[:m3.start()] + (
        f'{ind}b.loc[b.fight_id == r.fight_id, "meth_pick"] = meth[0] if meth else None\n'
        f'{ind}b.loc[b.fight_id == r.fight_id, "meth_p"] = meth[1] if meth else np.nan\n'
        f'{ind}b.loc[b.fight_id == r.fight_id, "meth_base"] = meth[2] if meth else np.nan\n'
    ) + out[m3.start():]

    # 6. rolling grade for the method market
    m4 = re.search(r"[ ]+w = dec\.tail\(200\)\n", out)
    if not m4:
        print("FAIL: could not locate the rolling-grade tail"); return 1
    out = out[:m4.start()] + GRADE + out[m4.start():]

    open(path, "w", encoding="utf-8").write(out)
    print(f"patched {path}  (backup at {path}.bak)")
    print("\nNOTE: meth_hit still needs wiring into grade(). The method pick names a")
    print("fighter AND a method, so it needs the result method, not just the winner.")
    print("Your grade() already returns (result, went_dist); extend it to also return")
    print("whether the posted meth_pick matched. Everything else is live.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    p = sys.argv[1]
    if "--revert" in sys.argv:
        if os.path.exists(p + ".bak"):
            shutil.move(p + ".bak", p); print("reverted"); sys.exit(0)
        print("no .bak found"); sys.exit(1)
    sys.exit(patch(p))
