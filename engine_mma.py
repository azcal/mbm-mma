"""
MBM MMA engine -- standalone, no market inputs.

Data: Wikipedia {{MMAevent bout}} templates across UFC, Bellator, PFL and ONE, plus
fighter infoboxes for date of birth. Non-UFC bouts update fighter states but never enter
evaluation, the same pattern as FCS opponents in NCAAF and Bo1/Bo5 in CS2.

Orientation is alphabetical by normalised name, so the base rate is 0.500 and the model
inherits nothing from a listing convention. ESPN's red-corner ordering was measured at
0.5353, worth 0.0025 log loss, and rejected as not worth a second data dependency.

Two heads:
  H1  P(A wins)                     -> moneyline
  H2  P(method | winner), 3-way     -> KO/TKO, submission, decision
Distance derives as 1 - [P(A)*P(finish|A) + P(B)*P(finish|B)], so the distance number can
never contradict the method numbers.

Age is the only fighter attribute that earned its place: +0.0082 log loss at t=3.43 on the
development window. Reach, height and stance measured at zero and are excluded.
"""
import json, re, unicodedata
import numpy as np, pandas as pd
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def norm_name(s):
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"\s*\(.*?\)\s*", " ", s)
    s = re.sub(r"[^a-zA-Z ]", " ", s).lower()
    return re.sub(r"\s+", " ", s).strip()


class Decay:
    __slots__ = ("d", "sw", "sx")
    def __init__(self, hl): self.d = 0.5 ** (1.0 / hl); self.sw = 0.0; self.sx = 0.0
    def add(self, x, w=1.0): self.sw = self.sw * self.d + w; self.sx = self.sx * self.d + w * x
    def rate(self, tau): return self.sx / (self.sw + tau)


WIN = ["elo_d", "n_a", "n_b", "deb_a", "deb_b", "lay_a", "lay_b",
       "a_kow", "a_subw", "a_decw", "a_kol", "a_subl",
       "b_kow", "b_subw", "b_decw", "b_kol", "b_subl",
       "age_a", "age_b", "age_d", "age_miss", "old_a", "old_b"]
METH = ["w_kow", "w_subw", "w_decw", "l_kol", "l_subl",
        "w_age", "l_age", "elo_gap", "sched", "is_title", "hw"]


def prepare(F, attrs):
    """Orient fights alphabetically and attach the target columns."""
    F = F.copy()
    F["date"] = pd.to_datetime(F.date)
    F["n1"] = F.f1.map(norm_name); F["n2"] = F.f2.map(norm_name)
    F = F[(F.n1 != "") & (F.n2 != "") & (F.n1 != F.n2)]
    F["A"] = np.where(F.n1 < F.n2, F.n1, F.n2)
    F["B"] = np.where(F.n1 < F.n2, F.n2, F.n1)
    F["y"] = (F.A == F.n1).astype(int)
    F["wt"] = F.weight.fillna("").str.replace(r"\s*\(.*", "", regex=True).str.strip()
    F["hw"] = F.wt.str.contains("Heavyweight", case=False, na=False).astype(int)
    # A scheduled fight has no result. Its alphabetical orientation still produces a
    # valid-looking y, so it must be flagged or the state engine will update ratings
    # from a fabricated outcome before the fight has happened.
    F["pending"] = F["m"].eq("PENDING") if "m" in F.columns else False
    F["meth"] = np.where(F.pending, "PENDING",
                         np.where(F.ko, "KO", np.where(F["sub"], "SUB", "DEC")))
    return F.sort_values(["date", "event"]).reset_index(drop=True)


def build(F, attrs, cfg):
    """Sequential state pass. Every feature is knowable before the fight starts."""
    K, HL, TAU, FIN = cfg["K"], cfg["hl"], cfg["tau"], cfg["finish_bonus"]
    elo = defaultdict(lambda: 1500.0)
    nf = defaultdict(int); last = {}
    st = defaultdict(lambda: {k: Decay(HL) for k in
                              ("ko_w", "sub_w", "dec_w", "ko_l", "sub_l")})
    rows = []
    for r in F.itertuples():
        sa, sb = st[r.A], st[r.B]
        ga, gb = attrs.get(r.A, {}), attrs.get(r.B, {})

        def age(g):
            d = g.get("dob")
            if not isinstance(d, pd.Timestamp):
                return np.nan
            a = (r.date - d).days / 365.25
            return a if 15 <= a <= 55 else np.nan
        aa, ab = age(ga), age(gb)
        la = (r.date - last[r.A]).days if r.A in last else np.nan
        lb = (r.date - last[r.B]).days if r.B in last else np.nan
        both_age = aa == aa and ab == ab
        rows.append({
            "idx": r.Index,
            "elo_d": elo[r.A] - elo[r.B], "elo_gap": abs(elo[r.A] - elo[r.B]),
            "n_a": min(nf[r.A], 20), "n_b": min(nf[r.B], 20),
            "deb_a": int(nf[r.A] == 0), "deb_b": int(nf[r.B] == 0),
            "lay_a": min(la, 900) if la == la else 400,
            "lay_b": min(lb, 900) if lb == lb else 400,
            "age_a": aa if aa == aa else 30.0, "age_b": ab if ab == ab else 30.0,
            "age_d": (aa - ab) if both_age else 0.0, "age_miss": int(not both_age),
            "old_a": max(0.0, aa - 33) if aa == aa else 0.0,
            "old_b": max(0.0, ab - 33) if ab == ab else 0.0,
            "a_kow": sa["ko_w"].rate(TAU), "a_subw": sa["sub_w"].rate(TAU),
            "a_decw": sa["dec_w"].rate(TAU), "a_kol": sa["ko_l"].rate(TAU),
            "a_subl": sa["sub_l"].rate(TAU),
            "b_kow": sb["ko_w"].rate(TAU), "b_subw": sb["sub_w"].rate(TAU),
            "b_decw": sb["dec_w"].rate(TAU), "b_kol": sb["ko_l"].rate(TAU),
            "b_subl": sb["sub_l"].rate(TAU)})
        if getattr(r, "pending", False) or r.y != r.y or not np.isfinite(r.y):
            continue
        w, l = (r.A, r.B) if r.y == 1 else (r.B, r.A)
        e = 1.0 / (1.0 + 10 ** (-(elo[w] - elo[l]) / 400.0))
        d = K * (FIN if not r.distance else 1.0) * (1.0 - e)
        elo[w] += d; elo[l] -= d
        key = "ko" if r.ko else ("sub" if getattr(r, "sub") else "dec")
        for k in ("ko", "sub", "dec"):
            st[w][f"{k}_w"].add(1.0 if k == key else 0.0)
        if key != "dec":
            for k in ("ko", "sub"):
                st[l][f"{k}_l"].add(1.0 if k == key else 0.0)
        nf[r.A] += 1; nf[r.B] += 1
        last[r.A] = last[r.B] = r.date
    return F.join(pd.DataFrame(rows).set_index("idx"))


def orient(D, winner):
    """Method-head features written from the point of view of a hypothetical winner."""
    p, o = ("a", "b") if winner == "A" else ("b", "a")
    return pd.DataFrame({
        "w_kow": D[f"{p}_kow"], "w_subw": D[f"{p}_subw"], "w_decw": D[f"{p}_decw"],
        "l_kol": D[f"{o}_kol"], "l_subl": D[f"{o}_subl"],
        "w_age": D[f"age_{p}"], "l_age": D[f"age_{o}"],
        "elo_gap": D.elo_gap, "sched": D.sched,
        "is_title": D.is_title.astype(int), "hw": D.hw}, index=D.index)


def fit_heads(D, tr, C):
    tr = tr & (~D.meth.eq("PENDING"))
    scw = StandardScaler().fit(D.loc[tr, WIN].values)
    mw = LogisticRegression(C=C, max_iter=3000).fit(
        scw.transform(D.loc[tr, WIN].values), D.loc[tr, "y"].values)
    Xa, Xb = orient(D[tr], "A"), orient(D[tr], "B")
    Xtr = Xa.where(D.loc[tr, "y"].eq(1), Xb)
    scm = StandardScaler().fit(Xtr[METH].values)
    mm = LogisticRegression(C=C, max_iter=3000).fit(
        scm.transform(Xtr[METH].values), D.loc[tr, "meth"].values)
    return (mw, scw, mm, scm)


def predict(heads, D):
    """Returns p_win (A), p_distance, and the six method probabilities."""
    mw, scw, mm, scm = heads
    pa = mw.predict_proba(scw.transform(D[WIN].values))[:, 1]
    cl = list(mm.classes_)
    PA = mm.predict_proba(scm.transform(orient(D, "A")[METH].values))
    PB = mm.predict_proba(scm.transform(orient(D, "B")[METH].values))
    iK, iS, iD = cl.index("KO"), cl.index("SUB"), cl.index("DEC")
    out = pd.DataFrame(index=D.index)
    out["p_win_a"] = pa
    for tag, P, w in (("a", PA, pa), ("b", PB, 1 - pa)):
        out[f"p_{tag}_ko"] = w * P[:, iK]
        out[f"p_{tag}_sub"] = w * P[:, iS]
        out[f"p_{tag}_dec"] = w * P[:, iD]
    out["p_distance"] = out.p_a_dec + out.p_b_dec
    return out


def load_attrs(path):
    A = pd.read_csv(path)
    A["dob"] = pd.to_datetime(A.dob, errors="coerce")
    return {r.key: {"dob": r.dob} for r in A.itertuples()}
