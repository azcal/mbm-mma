"""
MBM MMA daily runner (rebuild v1, frozen spec).
- Refreshes finished bouts from Wikipedia event pages (results, method, round, time)
- Pulls upcoming UFC cards from ESPN, the only source that lists fights before they happen
- Boards three markets: moneyline, goes-the-distance, and winner by method
- Grades logged forecasts, prints rolling metrics, last-200 tripwire
- Telegram push routed through tg_delta so refreshes only report what moved
Usage: python daily_mma.py [--date YYYY-MM-DD]
"""
import argparse, json, math, os, re, sys, time, datetime as dt
import numpy as np, pandas as pd, requests

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine_mma as E
import tg_delta

WIKI = "https://en.wikipedia.org/w/api.php"
ESPN = "https://site.api.espn.com/apis/site/v2/sports/mma/ufc/scoreboard"
WH = {"User-Agent": "MBM-forecasting/1.0 (contact: maparsons4@gmail.com)"}
EH = {"User-Agent": "Mozilla/5.0"}
FIGHTS = os.path.join(HERE, "data", "fights_clean.csv")
ATTRS = os.path.join(HERE, "data", "attrs.csv")
LOGS = os.path.join(HERE, "logs")
SPEC = json.load(open(os.path.join(HERE, "spec.json")))
CFG, C = SPEC["cfg"], SPEC["C"]
TRAIN_THROUGH = SPEC["train_through"]
HORIZON_DAYS = 14
MIN_EDGE = 0.05


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
    """ESPN is the only feed that lists fights before they happen."""
    lo = today.strftime("%Y%m%d")
    hi = (today + dt.timedelta(days=HORIZON_DAYS)).strftime("%Y%m%d")
    try:
        r = requests.get(ESPN, headers=EH, params={"dates": f"{lo}-{hi}"}, timeout=60).json()
    except Exception as ex:
        print("ESPN feed unavailable:", type(ex).__name__)
        return pd.DataFrame()
    rows = []
    for ev in r.get("events", []):
        for c in ev.get("competitions", []):
            cs = c.get("competitors", [])
            if len(cs) != 2 or c.get("status", {}).get("type", {}).get("completed"):
                continue
            names = [x.get("athlete", {}).get("displayName") for x in cs]
            if not all(names):
                continue
            rows.append({"fight_id": str(c.get("id")), "event": ev.get("name"),
                         "date": pd.Timestamp(ev.get("date")).tz_localize(None).normalize(),
                         "weight": c.get("type", {}).get("abbreviation", ""),
                         "f1": names[0], "f2": names[1],
                         "sched": c.get("format", {}).get("regulation", {}).get("periods", 3)})
    return pd.DataFrame(rows)


def run(today_str):
    today = dt.datetime.strptime(today_str, "%Y-%m-%d").date()
    preflight()
    F = refresh_results()
    A = E.load_attrs(ATTRS)
    up = fetch_upcoming(today)
    print(f"upcoming fights in {HORIZON_DAYS}d: {len(up)}")

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

    alerts, rows = [], []
    os.makedirs(LOGS, exist_ok=True)
    pred_path = os.path.join(LOGS, "predictions.csv")
    log = pd.read_csv(pred_path, dtype={"fight_id": str}) if os.path.exists(pred_path) else \
        pd.DataFrame(columns=["date", "fight_id", "event", "A", "B", "p_win_a", "p_elo",
                              "p_distance", "p_a_ko", "p_a_sub", "p_a_dec",
                              "p_b_ko", "p_b_sub", "p_b_dec", "model"])

    board = Dall[Dall.pending].copy()
    if len(board):
        pr = E.predict(heads, board)
        b = board[["fight_id", "event", "wt", "sched"]].copy()
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
        b["model"] = "v1"
        print(f"\n=== MBM MMA board ({len(b)} fights) ===")
        for _, r in b.sort_values(["date", "event"]).iterrows():
            fav, pf = (r.A, r.p_win_a) if r.p_win_a >= .5 else (r.B, 1-r.p_win_a)
            print(f"{r.date} {r.A} vs {r.B} ({r.wt}, {int(r.sched)}r)")
            print(f"    ML   {fav} {pf:.3f}  -> needs {min_odds(pf):.2f}")
            print(f"    DIST goes the distance {r.p_distance:.3f} -> needs {min_odds(r.p_distance):.2f}"
                  f"   |   inside {1-r.p_distance:.3f} -> needs {min_odds(1-r.p_distance):.2f}")
            best = max([(r.p_a_ko, f"{r.A} by KO/TKO"), (r.p_a_sub, f"{r.A} by sub"),
                        (r.p_a_dec, f"{r.A} by decision"), (r.p_b_ko, f"{r.B} by KO/TKO"),
                        (r.p_b_sub, f"{r.B} by sub"), (r.p_b_dec, f"{r.B} by decision")])
            print(f"    METH most likely: {best[1]} {best[0]:.3f} -> needs {min_odds(best[0]):.2f}")
            rows.append({"line": f"{r.A} vs {r.B}: {fav} {pf:.3f} >= {min_odds(pf):.2f} | "
                                 f"dist {r.p_distance:.3f} >= {min_odds(r.p_distance):.2f}",
                         "min_odds": min_odds(pf), "fight_id": r.fight_id})
        keep = [c for c in log.columns if c in b.columns]
        log = pd.concat([log[~log.fight_id.isin(b.fight_id)], b[keep]], ignore_index=True)
    else:
        print("no upcoming fights in horizon")

    # ---- grade ----
    res = F.assign(k=F.f1.map(E.norm_name) + "|" + F.f2.map(E.norm_name))
    won = {}
    for r in res.itertuples():
        a, bb = sorted([E.norm_name(r.f1), E.norm_name(r.f2)])
        won[f"{a}|{bb}"] = (1.0 if E.norm_name(r.f1) == a else 0.0, bool(r.distance))
    def grade(row):
        a, bb = sorted([E.norm_name(row.A), E.norm_name(row.B)])
        return won.get(f"{a}|{bb}", (np.nan, np.nan))
    g = log.apply(grade, axis=1, result_type="expand")
    log["result"], log["went_dist"] = g[0], g[1]
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

    tg, ch = os.environ.get("TG_TOKEN"), os.environ.get("TG_CHAT")
    if tg and ch:
        push = {r["fight_id"]: {"line": r["line"], "min_odds": r["min_odds"]} for r in rows}
        mode, lines = tg_delta.gate(LOGS, today.strftime("%Y%m%d"), push)
        if lines or alerts:
            txt = "\n".join(lines) if lines else "no price changes since the last push"
            if mode == "delta" and lines:
                txt = "changes since the last push:\n" + txt
            if alerts:
                txt += "\n\u26a0 TRIPWIRE: " + ", ".join(alerts) + " below base rate, stop betting this board"
            requests.post(f"https://api.telegram.org/bot{tg}/sendMessage",
                          data={"chat_id": ch, "text": f"MBM MMA\n{txt}"}, timeout=20)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=dt.date.today().isoformat())
    run(ap.parse_args().date)
