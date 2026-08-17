"""
MBM MMA daily runner (rebuild v1, frozen spec).

- Refreshes finished bouts from Wikipedia event pages (results, method, round, time)
- Pulls upcoming UFC cards from ESPN, the only source that lists fights before they happen
- Boards three markets: moneyline, goes-the-distance, and winner by method
- Grades logged forecasts, prints rolling metrics, last-200 tripwire
- Discord push routed through tg_delta so refreshes only report what moved

Usage: python daily_mma.py [--date YYYY-MM-DD]
"""
import argparse, difflib, json, math, os, re, sys, time, datetime as dt
from collections import defaultdict

import numpy as np, pandas as pd, requests

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine_mma as E
import tg_delta
import board_format as BF
import notify
import freshness as FR
import mbm_checks as CHK

# A stale copy of a shared file shows up as a wrong-looking board or an AttributeError
# halfway through a run. This turns it into a named failure in the workflow's import
# guard, which takes seconds. A minimum rather than an exact match, so a later
# improvement to a shared file does not force an edit here; raise it only for a change
# this runner depends on.
# freshness.py and mbm_checks.py are in the guard because this runner now wires the
# freshness guard and the check harness; the FR.guard(reporter=) call needs the
# 2026-08-09 freshness at runtime.
SHARED_MIN = "2026-08-02"   # mma_board two-row form; this runner does not call line()
for _mod in (BF, notify, tg_delta, FR, CHK):
    _have = getattr(_mod, "VERSION", "")
    if _have < SHARED_MIN:
        raise ImportError(
            f"{_mod.__name__}.py is version {_have or 'unstamped'}, this runner needs "
            f"{SHARED_MIN} or newer. Copy the current shared files into this repo.")

WIKI = "https://en.wikipedia.org/w/api.php"
ESPN = "https://site.api.espn.com/apis/site/v2/sports/mma/ufc/scoreboard"
WH = {"User-Agent": "MBM-forecasting/1.0 (contact: maparsons4@gmail.com)"}  # Wikipedia wants a UA
# ESPN 403s ANY custom User-Agent and only serves the requests default. Measured live
# 2026-08-10: this exact scoreboard URL with no headers returns 200, with "Mozilla/5.0"
# returns 403. The board sent a UA and every slate fetch failed silently (403 -> empty ->
# read as "no fights"), which no_silent_empty finally surfaced. Same reason nba/ncaaf carry
# ESPN_H = {}. Do NOT "fix" this by adding a UA.
EH = {}
FIGHTS = os.path.join(HERE, "data", "fights_clean.csv")
ATTRS = os.path.join(HERE, "data", "attrs.csv")
LOGS = os.path.join(HERE, "logs")
SPEC = json.load(open(os.path.join(HERE, "spec.json")))
CFG, C = SPEC["cfg"], SPEC["C"]
TRAIN_THROUGH = SPEC["train_through"]

# Cards are announced weeks out but only firm up in the last few days, and everything
# beyond the imminent card is noise on a board pushed daily. Five days covers Friday
# evening through the following Wednesday from any run.
HORIZON_DAYS = 5

# How near a result must sit to a logged fight's date to be its result. Rematches of the
# same pair are months apart, so this cleanly separates them; a fight and its own result
# share a date to within a day.
REMATCH_TOL_DAYS = 45

# Input-adequacy floor, NOT a probability band. A market priced at a coin flip means two
# opposite things: a genuine 50/50 between established fighters (a real forecast, worth
# publishing) or "no data" for debutants (the prior returned unchanged). The probability
# alone cannot tell them apart, so withhold on the INPUT: a fight where either fighter's
# recorded bout count is below this is not boarded. min(n_a, n_b) < 3 removes a Contender
# Series card of debutants while keeping every fight with a real record, including the true
# coin flips. Same shape as the engines' shrink, applied where the thinness is still
# measurable. Withheld fights are still forecast, logged and graded (only held from Discord).
ADEQ_MIN_BOUTS = 3

# Closeness band: a DISPLAY rule, not suppression, and MMA-specific. On a genuine even fight
# between two fighters with real records the model's separation is within noise, and the side
# named is only ESPN's competitor order (f1 = names[0]), so "A 50.3%" is a coin toss reported
# as a lean. Within this band the winner market renders "too close to call" -- no side, no
# price; the order-independent distance market still shows, and p_win_a is still logged and
# graded (the closeness decision itself is logged too, so the dashboard can report how often
# the model declines to pick). Does NOT port to WNBA/soccer, where A/home is a real side.
#
# 0.05 is a CURRENT ESTIMATE from the graded record, not a settled constant: below 0.55 no
# cut's hit rate is distinguishable from a coin flip (0.52-0.54 came in at 48.1% while
# 0.50-0.52 hit 57.4% -- the non-monotonicity is the tell), and everything from 0.60 up
# behaves (66.7% vs 64.4% expected). n is only ~100 below 0.55 and thin. Re-measure after
# another season.
CLOSE_BAND = 0.05

# Tracks board_format rather than restating it. This is the console-side copy, and a
# local 0.05 here while board_format ran at 0.00 meant the Actions log and Discord
# quoted different prices for the same game.
MIN_EDGE = BF.MIN_EDGE


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
    """Return (pick_or_None, best_lift). pick is (label, p, base, lift)."""
    P = np.array([r.p_a_ko, r.p_a_sub, r.p_a_dec,
                  r.p_b_ko, r.p_b_sub, r.p_b_dec], dtype=float)
    s = P.sum()
    if not np.isfinite(s) or s <= 0:
        return None, 0.0
    P = P / s
    pa = float(r.p_win_a)
    ko, sb, dc = base_tbl.get((r.wt, int(r.sched)), glob)
    B = np.array([ko * pa, sb * pa, dc * pa,
                  ko * (1 - pa), sb * (1 - pa), dc * (1 - pa)], dtype=float)
    B = B / B.sum()
    lift = np.where((P >= METH_LO) & (P <= METH_HI), P / np.maximum(B, 1e-9), -1.0)
    j = int(lift.argmax())
    best = float(max(lift.max(), 0.0))
    if lift[j] < lift_min:
        return None, best
    cls = METH_CLASSES[j]
    who = r.A if cls.startswith("A_") else r.B
    return (f"{who} by {cls.split('_')[1]}", float(P[j]), float(B[j]), float(lift[j])), best
# --- end MBM method market ---


def preflight():
    """Fail loudly and usefully if a committed data file is missing. The import guard in
    the workflow cannot see this, because these are read at run time, not import time."""
    need = {FIGHTS: "data/fights_clean.csv", ATTRS: "data/attrs.csv",
            os.path.join(HERE, "spec.json"): "spec.json"}
    missing = [name for path, name in need.items() if not os.path.exists(path)]
    if missing:
        print("MISSING REQUIRED FILES:")
        for m in missing:
            print(f"  {m}")
        print("\nThese ship with the repo and are not generated at run time. If you")
        print("uploaded through the GitHub web interface, select the folders rather than")
        print("individual files, or the data/ subfolder is dropped silently.")
        print(f"\nrepo root currently contains: {sorted(os.listdir(HERE))}")
        sys.exit(1)


def min_odds(p):
    p = min(max(float(p), 0.01), 0.99)
    return math.ceil((1 + MIN_EDGE) / p * 100) / 100.0


# ------------------------------------------------------------------ refresh
def refresh_results(days_back=75):
    """Re-scrape recent Wikipedia events so newly finished cards get graded."""
    F = pd.read_csv(FIGHTS)
    F["date"] = pd.to_datetime(F.date)
    cutoff = pd.Timestamp(dt.date.today()) - pd.Timedelta(days=days_back)
    try:
        from parse_mma import find_templates, split_fields, clean
    except Exception:
        print("parser module missing, skipping results refresh")
        return F
    r = requests.get(WIKI, headers=WH, params={
        "action": "query", "prop": "revisions", "rvprop": "content", "rvslots": "main",
        "titles": "List of UFC events", "redirects": 1, "format": "json"}, timeout=60).json()
    idx = list(r["query"]["pages"].values())[0]["revisions"][0]["slots"]["main"]["*"]
    MON = {m: i for i, m in enumerate(
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}
    recent = []
    for t, y, mon, d in re.findall(
            r"\|\s*\[\[([^\]|#]+?)(?:\|[^\]]*)?\]\]\s*\n\|\s*\{\{dts\|(\d{4})\|([A-Za-z]{3})\w*\|(\d{1,2})", idx):
        if mon[:3] not in MON:
            continue
        when = pd.Timestamp(int(y), MON[mon[:3]], int(d))
        if when >= cutoff and when <= pd.Timestamp(dt.date.today()):
            recent.append((t.strip(), when))
    if not recent:
        print("no recent events to refresh")
        return F
    q = requests.get(WIKI, headers=WH, params={
        "action": "query", "prop": "revisions", "rvprop": "content", "rvslots": "main",
        "titles": "|".join(t for t, _ in recent), "redirects": 1, "format": "json"}, timeout=90).json()
    pages = {p["title"]: p["revisions"][0]["slots"]["main"]["*"]
             for p in q.get("query", {}).get("pages", {}).values() if "revisions" in p}
    new = []
    for title, when in recent:
        txt = next((v for k, v in pages.items() if k.lower() == title.lower()), "")
        for body in find_templates(txt):
            f = [x.strip() for x in split_fields(body)]
            if f and f[0].lower().replace(" ", "").startswith("mmaeventbout"):
                f = f[1:]
            if len(f) < 7 or f[2].strip().lower().rstrip(".") not in ("def", "vs"):
                continue
            new.append({"weight": clean(f[0]), "f1": clean(f[1]),
                        "sep": f[2].strip().lower().rstrip("."), "f2": clean(f[3]),
                        "method": clean(f[4]), "round": clean(f[5]), "time": clean(f[6]),
                        "notes": clean(f[7]) if len(f) > 7 else "",
                        "event": title, "promotion": "UFC", "date": when})
    if not new:
        return F
    N = pd.DataFrame(new)
    N["m"] = N.method.map(_method)
    N = N[N.sep.eq("def") & N.m.isin(["KO", "TKO", "SUB", "DEC"])].copy()
    N["ko"] = N.m.isin(["KO", "TKO"]); N["sub"] = N.m.eq("SUB")
    N["distance"] = N.m.eq("DEC")
    N["rnd"] = pd.to_numeric(N["round"], errors="coerce")
    N["is_title"] = N.notes.fillna("").str.contains("[Cc]hampionship|[Tt]itle", regex=True)
    N["sched"] = np.where(N.m.eq("DEC") & N.rnd.notna(), N.rnd, 3)
    key = lambda d: d.event.astype(str) + "|" + d.f1.astype(str) + "|" + d.f2.astype(str)
    F = F[~key(F).isin(set(key(N)))]
    F = pd.concat([F, N], ignore_index=True).sort_values("date").reset_index(drop=True)
    F.to_csv(FIGHTS, index=False)
    print(f"results refreshed: +{len(N)} bouts from {len(recent)} recent events")
    return F


def _method(s):
    t = str(s).lower()
    for k, v in (("no contest", "NC"), ("draw", "Draw"), ("technical submission", "SUB"),
                 ("submission", "SUB"), ("tko", "TKO"), ("knockout", "KO"), ("decision", "DEC")):
        if k in t:
            return v
    return "KO" if t.startswith("ko") else "OTHER"


def fetch_upcoming(today):
    """ESPN is the only feed that lists fights before they happen.

    Returns (frame, feed_ok). feed_ok is False when the pull failed, so the caller can tell
    an ESPN outage (empty because we could not ask) from a genuinely empty card. Without it,
    "ESPN feed unavailable" followed by "0 fights" is indistinguishable from a quiet week --
    incident 3, the swallowed-error empty.
    """
    lo = today.strftime("%Y%m%d")
    hi = (today + dt.timedelta(days=HORIZON_DAYS)).strftime("%Y%m%d")
    try:
        resp = requests.get(ESPN, headers=EH, params={"dates": f"{lo}-{hi}"}, timeout=60)
        resp.raise_for_status()
        r = resp.json()
    except Exception as ex:
        print("ESPN feed unavailable:", type(ex).__name__)
        return pd.DataFrame(), False
    rows = []
    for ev in r.get("events", []):
        for c in ev.get("competitions", []):
            cs = c.get("competitors", [])
            # "pre" only. completed=False is also true of a bout currently being fought.
            state = (c.get("status", {}).get("type", {}) or {}).get("state")
            if len(cs) != 2 or state != "pre":
                continue
            names = [x.get("athlete", {}).get("displayName") for x in cs]
            if not all(names):
                continue
            rows.append({"fight_id": str(c.get("id")), "event": ev.get("name"),
                         "date": pd.Timestamp(ev.get("date")).tz_localize(None).normalize(),
                         "weight": c.get("type", {}).get("abbreviation", ""),
                         "f1": names[0], "f2": names[1],
                         # carried so pregame_only can confirm the filter held
                         "state": state,
                         "sched": c.get("format", {}).get("regulation", {}).get("periods", 3)})
    return pd.DataFrame(rows), True


def run(today_str):
    today = dt.datetime.strptime(today_str, "%Y-%m-%d").date()
    chk = CHK.Reporter("MMA", today.strftime("%Y%m%d"))
    preflight()
    F = refresh_results()
    A = E.load_attrs(ATTRS)

    # Freshness. MMA never wired this before: freshness.py sat in the repo unused, so the
    # board priced fights with no check that the Wikipedia results history was current.
    # ESPN's mma/ufc scoreboard lists completed cards, so probe it against the newest bout
    # the history holds. grace 40: measured 295 UFC event days since 2022 -- median gap 6,
    # but the holiday break after 2025-12-20 ran 35 days with 29 gaps over a week, so a
    # shorter grace would block a healthy board on a throttled probe in late December.
    # season_types=None because UFC events do not carry the 1/2/3 US season convention.
    # reporter=chk registers the guard, which is what require("freshness") verifies ran.
    #
    # exclude: ESPN files Dana White's Contender Series under mma/ufc, but refresh_results
    # scrapes Wikipedia's "List of UFC events", which does not include it -- measured, zero
    # Contender Series bouts among 10,216 in the history. So every DWCS card made the probe
    # report the board stale against an event it can never ingest, and MMA blocked every
    # Tuesday of the DWCS season with no refresh able to clear it. Same job season_types
    # does for NFL preseason: a probe must count only what the history is able to contain.
    _newest = pd.to_datetime(F.date).max().date() if len(F) else None
    FR.guard("MMA", _newest, "mma/ufc", grace_days=40, season_types=None,
             exclude=r"Contender Series", reporter=chk)

    up, _feed_ok = fetch_upcoming(today)
    print(f"upcoming fights in {HORIZON_DAYS}d: {len(up)}")
    # Incident 3 live: a failed ESPN pull returns empty, which reads exactly like a quiet
    # week. WARN, not FAIL -- a slate outage means no board this run, but must not kill the
    # independent Wikipedia-based grading below.
    chk.no_silent_empty("espn_slate", is_empty=(len(up) == 0), call_failed=not _feed_ok,
                        warn_only=True)

    pseudo = []
    if len(up):
        for r in up.itertuples():
            pseudo.append({"weight": r.weight, "f1": r.f1, "f2": r.f2, "sep": "def",
                           "method": "", "round": np.nan, "time": "", "notes": "",
                           "event": r.event, "promotion": "UFC", "date": r.date,
                           "m": "PENDING", "ko": False, "sub": False, "distance": False,
                           "is_title": False, "sched": r.sched, "fight_id": r.fight_id})
    G = pd.concat([F, pd.DataFrame(pseudo)], ignore_index=True) if pseudo else F
    G["is_title"] = G.is_title.fillna(False)
    G["sched"] = pd.to_numeric(G.sched, errors="coerce").fillna(3)
    P = E.prepare(G, A)
    # One pass. prepare() flags pending rows and build() skips their state updates, so
    # scheduled fights read the current ratings without contributing to them.
    Dall = E.build(P, A, CFG)
    tr = (Dall.date.dt.year <= TRAIN_THROUGH) & (~Dall.pending)
    heads = E.fit_heads(Dall, tr, C)
    _meth_base, _meth_glob = _meth_base_rates(Dall, tr)

    alerts, rows = [], []
    os.makedirs(LOGS, exist_ok=True)
    pred_path = os.path.join(LOGS, "predictions.csv")
    log = pd.read_csv(pred_path, dtype={"fight_id": str}) if os.path.exists(pred_path) else \
        pd.DataFrame(columns=["date", "fight_id", "event", "A", "B", "p_win_a", "p_elo",
                              "p_distance", "p_a_ko", "p_a_sub", "p_a_dec",
                              "p_b_ko", "p_b_sub", "p_b_dec", "model", "close"])
    if "close" not in log.columns:
        log["close"] = pd.NA   # rows logged before the closeness rule existed

    # Pregame guard. fetch_upcoming already filters to state=="pre"; this confirms the
    # filter held, and (the arm the filter cannot cover) refuses to re-board a fight the
    # log has already graded. MMA history carries no ESPN id, so the history arm keys on
    # the graded log's fight_ids rather than the results store.
    _graded_ids = (set(log.loc[log.result.notna(), "fight_id"].astype(str))
                   if "result" in log.columns else set())
    chk.pregame_only(up.to_dict("records") if len(up) else [],
                     history_ids=_graded_ids, id_key="fight_id")

    board = Dall[Dall.pending].copy()
    if len(board):
        pr = E.predict(heads, board)
        b = board[["fight_id", "event", "wt", "sched"]].copy()
        # Recorded bout counts, carried for the input-adequacy floor below. engine_mma holds
        # them as win-head features, capped at 20.
        b["n_a"] = board.n_a.values
        b["n_b"] = board.n_b.values
        # Title-cased for the board and the dashboard; grading re-normalises, so either
        # casing joins correctly.
        b["A"] = board.A.str.title().values
        b["B"] = board.B.str.title().values
        # The Elo floor is what the dashboard grades the model against. Without it the
        # MMA tab has no baseline to draw.
        b["p_elo"] = 1 / (1 + 10 ** (-board.elo_d.values / 400))
        b["date"] = board.date.dt.strftime("%Y-%m-%d")
        for c in pr.columns:
            b[c] = pr[c].values
        # Closeness is decided on p_win_a and logged, so the dashboard can later report how
        # often the model declines to name a side.
        b["close"] = (b["p_win_a"] - 0.5).abs() < CLOSE_BAND
        # Tag from the spec, not a literal, so bumping the head without updating the spec
        # trips model_identity below.
        b["model"] = SPEC.get("model", "v1")
        for _, r in b.sort_values(["date", "event"]).iterrows():
            fav, pf = (r.A, r.p_win_a) if r.p_win_a >= .5 else (r.B, 1-r.p_win_a)
            n_min = int(min(r.n_a, r.n_b))
            # Closeness only applies to fights that clear the adequacy floor; a no-data fight
            # is withheld entirely, not shown as "too close to call".
            close_display = bool(r.close) and n_min >= ADEQ_MIN_BOUTS
            thin_note = (f"   [WITHHELD: thin record, min {n_min} bout(s) < {ADEQ_MIN_BOUTS}]"
                         if n_min < ADEQ_MIN_BOUTS else "")
            print(f"{r.date}  {r.A} vs {r.B}  ({r.wt}, {int(r.sched)}r)")
            if close_display:
                print(f"  ML   too close to call (p_win_a {r.p_win_a:.3f}, within "
                      f"{CLOSE_BAND:.2f} of 0.50)")
            else:
                print(f"  ML   {fav} {pf:.3f}  -> needs {min_odds(pf):.2f}{thin_note}")
            print(f"  DIST goes the distance {r.p_distance:.3f} -> needs {min_odds(r.p_distance):.2f}"
                  f"  | inside {1-r.p_distance:.3f} -> needs {min_odds(1-r.p_distance):.2f}")
            meth, _mbest = _meth_pick(r, _meth_base, _meth_glob)
            if meth:
                _mlbl, _mp, _mb, _mlift = meth
                print(f"  METH {_mlbl} {_mp:.3f} -> needs {min_odds(_mp):.2f}"
                      f"  (base {_mb:.3f}, {_mlift:.2f}x)")
            else:
                # Always emit a line. Silence cannot be told apart from an unapplied patch.
                print(f"  METH no edge (best {_mbest:.2f}x, need {METH_LIFT_MIN:.2f}x)")
            combo = BF.mma_board(r.p_a_ko, r.p_a_sub, r.p_a_dec,
                                 r.p_b_ko, r.p_b_sub, r.p_b_dec, r.A, r.B)
            if close_display:
                # Replace the winner market with a no-side "too close to call", keep the
                # order-independent distance market. The distance line is the one naming
                # "goes the distance" or "ends inside"; the winner line never does.
                dist_lines = [ln for ln in combo
                              if "goes the distance" in ln or "ends inside" in ln]
                # Leading indent so notify renders it as subtext, not a plain-text line that
                # reads like a header.
                combo = ["  too close to call"] + dist_lines
            # close_display suppresses the WINNER market only. The method pick's lift is
            # computed against a base that also carries p_win_a, so the winner lean cancels
            # and the lift measures the method departure on its own. A coin-flip moneyline
            # does not make "A by SUB at 1.8x its base rate" meaningless.
            if meth and n_min >= ADEQ_MIN_BOUTS:
                combo = list(combo) + [
                    f"{_mlbl}  {_mp*100:.1f}%  find {1/_mp:.2f} / "
                    f"{BF.american(_mp)} or better"]
            for ln in combo:
                print("  " + ln)
            b.loc[b.fight_id == r.fight_id, "meth_pick"] = meth[0] if meth else None
            b.loc[b.fight_id == r.fight_id, "meth_p"] = meth[1] if meth else np.nan
            b.loc[b.fight_id == r.fight_id, "meth_base"] = meth[2] if meth else np.nan
            rows.append({"date": r.date, "game": f"{r.A} vs {r.B}",
                         "pick": ("too close to call" if close_display else fav),
                         "p": pf, "fight_id": r.fight_id, "combo": combo,
                         "n_min": n_min, "close": bool(r.close)})
        for _c in ("meth_pick", "meth_p", "meth_base"):
            if _c not in log.columns:
                log[_c] = np.nan
        keep = [c for c in log.columns if c in b.columns]
        log = pd.concat([log[~log.fight_id.isin(b.fight_id)], b[keep]], ignore_index=True)
    else:
        print("no upcoming fights in horizon")

    # ---- grade ----
    # Keyed on the sorted normalised pair, but as a LIST per pair with each result's date,
    # not a single overwrite. A pair that fought more than once (a rematch) shares one key,
    # and storing only the last result graded the earlier fight against the later winner:
    # measured at 7.1% of the history being repeat pairings, ~1 in 5 with a different
    # winner the second time, so the earlier leg silently graded wrong. Resolving by
    # nearest date within tolerance picks the right bout; a pair that appears once has no
    # ambiguity and resolves directly. The value stored is the WINNER'S name rather than a
    # flag, because `result` is paired with p_win_a = P(row.A wins), and a "did the
    # first-sorting name win" flag only agrees while row.A happens to sort first.
    won, keys = defaultdict(list), []
    for r in F.itertuples():
        a, bb = sorted([E.norm_name(r.f1), E.norm_name(r.f2)])
        k = f"{a}|{bb}"
        won[k].append((pd.to_datetime(r.date), E.norm_name(r.f1), bool(r.distance)))
        keys.append(k)

    def _resolve(k, row_date):
        """Winner and distance for pair-key `k` nearest the logged fight's date.

        One result on the key: unambiguous, return it. Several (a rematch): the nearest
        within REMATCH_TOL_DAYS, else None rather than a guess."""
        lst = won.get(k)
        if not lst:
            return None
        if len(lst) == 1:
            return lst[0][1], lst[0][2]
        rd = pd.to_datetime(row_date)
        d0, winner, dist = min(lst, key=lambda x: abs((x[0] - rd).days))
        if abs((d0 - rd).days) <= REMATCH_TOL_DAYS:
            return winner, dist
        return None

    def close_key(k):
        """Nearest result key, for a bout whose names differ between the two sources.

        The board takes names from ESPN and results from Wikipedia, and they disagree
        often enough to matter: Jessie against Jesse, Billy against Bill, a surname
        carrying an extra component on one side. An exact key misses all of those and the
        fight sits ungraded forever with nothing said.

        Requires a high similarity AND a clear gap to the runner-up, so a near-miss on one
        card cannot be resolved to a different bout.
        """
        best = difflib.get_close_matches(k, keys, n=2, cutoff=0.85)
        if not best:
            return None
        if len(best) > 1:
            r1 = difflib.SequenceMatcher(None, k, best[0]).ratio()
            r2 = difflib.SequenceMatcher(None, k, best[1]).ratio()
            if r1 - r2 < 0.05:
                return None
        return best[0]

    fuzzy = []

    def grade(row):
        a, bb = sorted([E.norm_name(row.A), E.norm_name(row.B)])
        k = f"{a}|{bb}"
        hit = _resolve(k, row.date)
        if hit is None:
            alt = close_key(k)
            if alt is None:
                return (np.nan, np.nan)
            hit = _resolve(alt, row.date)
            if hit is None:
                return (np.nan, np.nan)
            fuzzy.append(f"{row.A} vs {row.B} -> {alt.replace('|', ' vs ')}")
        winner, dist = hit
        # Compare the winner to row.A directly rather than to whichever name sorts first.
        na, nb = E.norm_name(row.A), E.norm_name(row.B)
        if winner == na:
            return (1.0, dist)
        if winner == nb:
            return (0.0, dist)
        # Matched fuzzily, so the winner string may be a variant of either name.
        return (1.0 if difflib.SequenceMatcher(None, winner, na).ratio()
                > difflib.SequenceMatcher(None, winner, nb).ratio() else 0.0, dist)

    g = log.apply(grade, axis=1, result_type="expand")
    if fuzzy:
        print(f"graded {len(fuzzy)} bout(s) by nearest name match, check these look right:")
        for f in fuzzy[:6]:
            print(f"  {f}")
    log["result"], log["went_dist"] = g[0], g[1]
    # Anything past its date and still ungraded. Silence here used to mean the bout simply
    # never appeared in the rolling metrics, with nothing to say why.
    _cut = (pd.Timestamp.now("UTC").tz_localize(None) - pd.Timedelta(days=2))
    _stale = log[log.result.isna() & (pd.to_datetime(log.date, errors="coerce") < _cut)]
    if len(_stale):
        print(f"{len(_stale)} bout(s) fought but still ungraded, names not matched to a "
              f"result:")
        for _, _r in _stale.sort_values("date").tail(8).iterrows():
            print(f"  {_r.date}  {_r.A} vs {_r.B}")

    # ---- invariant checks, before anything is persisted or published ----
    if len(board):
        chk.key_join("fight_id", log.fight_id, b.fight_id)
    chk.column_contract("result", log.result, "binary")
    chk.column_contract("went_dist", log.went_dist, "binary")
    chk.column_contract("p_win_a", log.p_win_a, "prob")
    # Two keys, two jobs: the board WRITES the single current head (SPEC["model"], a scalar
    # so it broadcasts to every new row); the check compares the log's tags against the
    # ACCEPTED SET (SPEC["model_tags"], e.g. ["v1", "v1-backfill"]). Using one key for both
    # is what caused the length-2-vs-17 write error and the false FAIL on v1-backfill.
    chk.model_identity(log.model.dropna().unique(),
                       SPEC.get("model_tags", [SPEC.get("model", "v1")]))
    if rows:
        chk.log_board_agreement(pushed_ids=[r["fight_id"] for r in rows],
                                logged_ids_today=log.fight_id)
    # Conviction backstop, on the fights that clear the adequacy floor. With the bout-count
    # rule in place, an identical-probability cluster here means a prior slipped through in
    # a way the adequacy rule did not anticipate -- exactly when you want to hear about it.
    chk.no_conviction([r["p"] for r in rows if r["n_min"] >= ADEQ_MIN_BOUTS])
    # Gate the write AND the push: a failed invariant aborts here, before the log is
    # persisted (duplicate/misgraded rows) and before Discord.
    chk.finish(strict=True, require=["freshness"])

    log.to_csv(pred_path, index=False)

    dec = log[log.result.isin([0.0, 1.0])]
    if len(dec) >= 30:
        y = dec.result.astype(int).values
        base = float(Dall.loc[tr, "y"].mean())
        print(f"\nrolling grade (n={len(y)}):")
        pc = np.clip(dec.p_win_a.values.astype(float), 1e-6, 1-1e-6)
        from sklearn.metrics import log_loss
        print(f"  moneyline: LL={log_loss(y, pc):.4f}  base={log_loss(y, np.full(len(y), base)):.4f}  "
              f"hit={np.mean((pc > .5) == (y == 1)):.3f}")
        dd = dec[dec.went_dist.notna()]
        if len(dd) >= 30:
            yd = dd.went_dist.astype(int).values
            pd_ = np.clip(dd.p_distance.values.astype(float), 1e-6, 1-1e-6)
            print(f"  distance : LL={log_loss(yd, pd_):.4f}  base="
                  f"{log_loss(yd, np.full(len(yd), yd.mean())):.4f}")
        # method market: graded only on rows where a pick was actually posted
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
        w = dec.tail(200)
        if len(w) >= 200:
            yy = w.result.astype(int).values
            edge = log_loss(yy, np.full(len(yy), base)) - \
                log_loss(yy, np.clip(w.p_win_a.values.astype(float), 1e-6, 1-1e-6))
            print(f"  last-200 edge vs base: {edge:+.4f}  "
                  f"[{'OK' if edge > 0 else 'TRIPWIRE: bench review'}]")
            if edge <= 0:
                alerts.append("MMA moneyline")
        dec.to_csv(os.path.join(LOGS, "graded.csv"), index=False)

    push, skipped, thin = {}, 0, 0
    if rows:
        for r in rows:
            # Input-adequacy floor: withhold a fight where either fighter's record is too
            # thin for the model to be doing more than returning its prior. A genuine coin
            # flip between two established fighters is a real forecast and still publishes.
            # Still forecast, logged and graded above -- this only holds it from Discord.
            if r["n_min"] < ADEQ_MIN_BOUTS:
                thin += 1
                continue
            if BF.min_odds(r["p"]) < BF.PRICE_FLOOR:
                skipped += 1
                continue
            # Three lines per fight: moneyline lean, distance lean, and the two as a
            # same-game parlay, which books post as one ticket.
            combo = r.get("combo") or []
            if not combo:
                skipped += 1
                continue
            # TWO spaces after the date: notify.HEADER is ^(date)\s{2,}(.+)$, so a single
            # space falls through to plain text and the fight titles lose their bold and the
            # blank-line separation, flattening the whole post.
            block = [f"{r['date']}  {r['game']}"] + combo
            push[r["fight_id"]] = {"line": "\n".join(block),
                                   # A too-close fight has no ML price; pin its gate value so
                                   # a p_win_a drifting inside the band does not emit a
                                   # phantom "price moved" note. Only a status change (into or
                                   # out of the band), carried by the match field, is news.
                                   "min_odds": (2.0 if r.get("close") else BF.min_odds(r["p"])),
                                   # Carries BOTH the matchup and the side. The matchup
                                   # alone catches a replaced opponent under the same ESPN
                                   # id but is blind to the model changing its mind: on a
                                   # near-even fight the side can flip while the price
                                   # moves less than the delta tolerance, so the channel
                                   # keeps showing the old fighter while the log holds the
                                   # new one. That is how a published pick and a graded
                                   # pick came apart on 2026-08-04.
                                   "match": f"{r['pick']} in {r['game']}"}
        print("\n=== MBM MMA board ===")
        for _v in push.values():
            print(_v["line"])
        if skipped or thin:
            _bits = []
            if thin:
                _bits.append(f"{thin} withheld on thin record (<{ADEQ_MIN_BOUTS} bouts)")
            if skipped:
                _bits.append(f"{skipped} below price floor "
                             f"{BF.PRICE_FLOOR:.2f} / {BF.american(BF.PRICE_FLOOR)}")
            print("(" + "; ".join(_bits) + ")")

    mode, lines = tg_delta.gate(LOGS, today.strftime("%Y%m%d"), push)
    if lines or alerts:
        body = list(lines) if lines else ["no price changes since the last push"]
        _foot_bits = []
        if thin:
            _foot_bits.append(f"{thin} withheld (thin record)")
        if skipped:
            _foot_bits.append(f"{skipped} below price floor "
                              f"{BF.PRICE_FLOOR:.2f} / {BF.american(BF.PRICE_FLOOR)}")
        foot = " · ".join(_foot_bits) if _foot_bits else None
        title = "MBM MMA" + (" · changes since the last push"
                             if mode == "delta" and lines else "")
        colour = notify.GREEN
        if alerts:
            body = body + ["", "TRIPWIRE: " + ", ".join(alerts)
                           + " below base rate, stop betting this board"]
            colour = notify.RED
        notify.send(title, body, foot, colour)
    elif not push:
        print("no board to send")
    else:
        print("board unchanged since the last push, nothing sent")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=dt.date.today().isoformat())
    run(ap.parse_args().date)
