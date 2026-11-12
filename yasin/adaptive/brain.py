"""
The bot's memory.

Brain   - everything learned, saved to data/brain.json and loaded on the next
          start. Counts persist, so features unlocked in an earlier session
          are available from the first second of the next one.
DataLog - raw CSV logs of every book, market trade, own fill and news post,
          one folder per session under data/raw/. Use them for offline analysis.
"""
import csv
import json
import math
import os
import time

import config as C
from pricing import clip, solve_linear

N_FEATURES = 4
FEATURE_NAMES = ["book_imbalance", "trade_flow", "deviation_from_model", "informed_flow"]


def ema(old, x, alpha):
    return x if old is None else (1 - alpha) * old + alpha * x


def halflife_alpha(n):
    return 1 - 0.5 ** (1.0 / max(1.0, n))


class Brain:
    VERSION = 1
    SCHEMA = 2      # 2 = history kept apart from live data, confidence weighting

    def __init__(self, data_dir=C.DATA_DIR):
        os.makedirs(data_dir, exist_ok=True)
        self.path = os.path.join(data_dir, "brain.json")
        self.s = {"version": self.VERSION, "schema": self.SCHEMA, "sessions": 0, "inst": {},
                  "names": {}, "reg": {}, "news": {"n": 0, "impact": None}, "bandit": {}}
        if os.path.exists(self.path):
            try:
                with open(self.path) as fh:
                    loaded = json.load(fh)
                if loaded.get("version") == self.VERSION:
                    self.s.update(loaded)
            except (OSError, ValueError) as e:
                print(f"brain.json unreadable ({e}), starting fresh")
        if self.s.get("schema") != self.SCHEMA:
            # Older brains mixed replayed history into the live estimates and the
            # prediction model used other features: drop those, keep fills/markouts/bandit.
            self.s["reg"] = {}
            self.s["pretrained"] = []
            self.s["schema"] = self.SCHEMA
        self._w_cache = {}

    def save(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(self.s, fh)
        os.replace(tmp, self.path)

    @staticmethod
    def conf(n, half):
        """Confidence 0..1 from an amount of evidence: half -> 50%."""
        return n / (n + half) if n > 0 else 0.0

    # ------------------------------------------------------------------ per instrument
    def inst(self, iid):
        st = self.s["inst"].setdefault(iid, {
            "n_books": 0, "sigma2": None, "n_vol": 0, "vol_method": 3,
            "sigma2_hist": None, "n_vol_hist": 0,
            "basis": 0.0, "n_basis": 0, "rel_err": None,
            "iv": None, "n_iv": 0, "fills": 0, "passive_fills": 0,
            "mk_bid": None, "mk_ask": None, "n_bid": 0, "n_ask": 0,
        })
        if st.get("vol_method") != 3:
            # earlier versions: per-loop vol (noisy) or history mixed into live counts
            st.update(sigma2=None, n_vol=0, vol_method=3, sigma2_hist=None, n_vol_hist=0,
                      n_books=0, basis=0.0, n_basis=0, rel_err=None)
        return st

    def _hist_weight(self, st):
        """History is a prior worth HIST_PRIOR_VOL_SAMPLES live samples, shrinking to 0
        when live vol disagrees with it (ratio HIST_DISAGREE_RATIO -> weight 0)."""
        if not st["sigma2_hist"]:
            return 0.0
        w = float(C.HIST_PRIOR_VOL_SAMPLES)
        if st["sigma2"] and st["n_vol"] >= 3:
            ratio = math.sqrt(st["sigma2"] / st["sigma2_hist"])
            w *= clip(1 - abs(math.log(ratio)) / math.log(C.HIST_DISAGREE_RATIO), 0.0, 1.0)
        return w

    def sigma_1s(self, iid):
        st = self.inst(iid)
        wh = self._hist_weight(st)
        nl = st["n_vol"] if st["sigma2"] else 0
        if wh + nl <= 0:
            return 0.0
        s2 = (wh * (st["sigma2_hist"] or 0.0) + nl * (st["sigma2"] or 0.0)) / (wh + nl)
        return math.sqrt(s2)

    def vol_conf(self, iid):
        """History counts half as much as live evidence (it speeds things up, it can't
        carry them): history alone stays below 50%."""
        st = self.inst(iid)
        return self.conf(st["n_vol"] + 0.5 * self._hist_weight(st), C.CONF_HALF_VOL)

    def vol_learned(self, iid):
        """>= 50% confidence AND at least half the evidence must be live."""
        return self.vol_conf(iid) >= 0.5 and self.inst(iid)["n_vol"] >= C.CONF_HALF_VOL / 2

    def set_hist_vol(self, iid, sigma2, n):
        st = self.inst(iid)
        st["sigma2_hist"], st["n_vol_hist"] = sigma2, n

    def count_book(self, iid):
        self.inst(iid)["n_books"] += 1

    def update_vol(self, iid, move, dt):
        """move = fair-value change over dt seconds (dt ~ VOL_SAMPLE_S). Live data only."""
        if dt <= 0:
            return
        st = self.inst(iid)
        a = 1 - 0.5 ** (dt / C.VOL_HALFLIFE_S)
        st["sigma2"] = ema(st["sigma2"], move * move / dt, a)
        st["n_vol"] += 1

    def update_basis(self, iid, market_minus_model, rel_err):
        st = self.inst(iid)
        st["basis"] = ema(st["basis"] if st["n_basis"] else None, market_minus_model,
                          halflife_alpha(C.BASIS_HALFLIFE))
        st["n_basis"] += 1
        st["rel_err"] = ema(st["rel_err"], rel_err, 0.02)

    def basis(self, iid):
        st = self.inst(iid)
        return self.conf(st["n_basis"], C.TIER1_BOOKS) * st["basis"]

    def model_trusted(self, iid):
        e = self.inst(iid)["rel_err"]
        return e is None or e < C.DISTRUST_REL

    def update_iv(self, iid, iv):
        st = self.inst(iid)
        st["iv"] = ema(st["iv"], iv, 0.05)
        st["n_iv"] += 1

    # ------------------------------------------------------------------ fills / adverse selection
    def record_markout(self, iid, side, markout_ticks):
        st = self.inst(iid)
        k = "bid" if side == "bid" else "ask"
        st["mk_" + k] = ema(st["mk_" + k], markout_ticks, halflife_alpha(30))
        st["n_" + k] += 1

    def adverse_extra_ticks(self, iid, side):
        """Widen the side that gets picked off, tighten the side that earns -- in
        proportion to how many live fills back it up (no on/off threshold)."""
        st = self.inst(iid)
        mk = st["mk_" + side]
        if mk is None:
            return 0.0
        return self.conf(st["n_" + side], C.CONF_HALF_FILLS) * 0.5 * clip(C.MARKOUT_TARGET_TICKS - mk, -1.0, 4.0)

    # ------------------------------------------------------------------ bandit on edge multiplier
    def bandit_unlocked(self, iid):
        return self.inst(iid)["fills"] >= C.TIER3_FILLS

    def bandit_choose(self, iid, rng):
        arms = self.s["bandit"].setdefault(iid, {})
        if not self.bandit_unlocked(iid):
            return 1.0
        stats = [(m, arms.get(str(m), {"n": 0, "mean": 0.0, "m2": 0.0})) for m in C.BANDIT_ARMS]
        unexplored = [m for m, a in stats if a["n"] < 3]
        if unexplored:
            return unexplored[0]
        best, best_v = 1.0, -1e18
        for m, a in stats:                         # Thompson sampling, normal approx
            var = a["m2"] / (a["n"] - 1) if a["n"] > 1 else 1.0
            v = a["mean"] + rng.gauss(0, 1) * math.sqrt(max(var, 1e-6) / a["n"])
            if v > best_v:
                best, best_v = m, v
        return best

    def bandit_record(self, iid, arm, reward):
        a = self.s["bandit"].setdefault(iid, {}).setdefault(str(arm), {"n": 0, "mean": 0.0, "m2": 0.0})
        a["n"] += 1                                # Welford
        d = reward - a["mean"]
        a["mean"] += d / a["n"]
        a["m2"] += d * (reward - a["mean"])

    # ------------------------------------------------------------------ counterparties
    def record_name(self, name, move_ticks):
        n = self.s["names"].setdefault(name, {"n": 0, "mk": None})
        n["mk"] = ema(n["mk"], move_ticks, 0.05)
        n["n"] += 1

    def name_score(self, name):
        n = self.s["names"].get(name)
        if not n or n["mk"] is None:
            return 0.0
        return self.conf(n["n"], C.CONF_HALF_NAMES) * clip(n["mk"], -3.0, 3.0)

    # ------------------------------------------------------------------ news
    def record_news_impact(self, ratio):
        nw = self.s["news"]
        nw["impact"] = ema(nw["impact"], ratio, 0.2)
        nw["n"] += 1

    def news_pause(self):
        nw = self.s["news"]
        if nw["n"] >= C.TIER2_NEWS and nw["impact"] is not None and nw["impact"] < 1.5:
            return 0.0                             # learned: news doesn't move prices much
        return C.NEWS_PAUSE_S

    # ------------------------------------------------------------------ short-term alpha (ridge)
    def _reg(self, key):
        r = self.s["reg"].setdefault(key, {
            "n": 0, "n_live": 0, "n_hist": 0, "XtX": [[0.0] * N_FEATURES for _ in range(N_FEATURES)],
            "Xty": [0.0] * N_FEATURES, "mse_model": None, "mse_zero": None})
        r.setdefault("n_live", 0)
        r.setdefault("n_hist", 0)
        return r

    def _weights(self, key):
        r = self._reg(key)
        cached = self._w_cache.get(key)
        if cached and r["n"] - cached[0] < 50:
            return cached[1]
        A = [[r["XtX"][i][j] + (C.RIDGE_LAMBDA if i == j else 0.0) for j in range(N_FEATURES)]
             for i in range(N_FEATURES)]
        w = solve_linear(A, r["Xty"]) or [0.0] * N_FEATURES
        self._w_cache[key] = (r["n"], w)
        return w

    def add_sample(self, key, x, y, weight=1.0, live=True):
        """Live samples first test the model out of sample (that test alone decides
        how much the model may act); history samples only help fit it."""
        r = self._reg(key)
        if live:
            pred = sum(a * b for a, b in zip(self._weights(key), x))
            r["mse_model"] = ema(r["mse_model"], (y - pred) ** 2, 0.01)
            r["mse_zero"] = ema(r["mse_zero"], y * y, 0.01)
            r["n_live"] += 1
        else:
            r["n_hist"] += 1
        lam = 0.5 ** (1.0 / C.RIDGE_HALFLIFE_SAMPLES)      # forget old data gradually
        for i in range(N_FEATURES):
            r["Xty"][i] = lam * r["Xty"][i] + weight * x[i] * y
            for j in range(N_FEATURES):
                r["XtX"][i][j] = lam * r["XtX"][i][j] + weight * x[i] * x[j]
        r["n"] += 1

    def alpha_scale(self, key):
        """0..1: live-proven improvement over 'no move' x amount of live evidence."""
        r = self.s["reg"].get(key)
        if not r or not r.get("n_live") or not r["mse_zero"]:
            return 0.0
        improvement = 1 - r["mse_model"] / r["mse_zero"]
        return (clip(improvement / C.ALPHA_FULL_IMPROVEMENT, 0.0, 1.0)
                * self.conf(r["n_live"], C.CONF_HALF_ALPHA))

    def alpha_active(self, key):
        return self.alpha_scale(key) > 0.0

    def predict(self, key, x):
        s = self.alpha_scale(key)
        if s <= 0:
            return 0.0
        return s * clip(sum(a * b for a, b in zip(self._weights(key), x)),
                        -C.ALPHA_CLIP_TICKS, C.ALPHA_CLIP_TICKS)

    # ------------------------------------------------------------------ display
    def tier(self, iid):
        """Display only: how far the live evidence has come. Nothing switches on it."""
        st = self.inst(iid)
        if st["fills"] >= C.TIER3_FILLS:
            return 3
        if st["passive_fills"] >= C.TIER2_FILLS:
            return 2
        if self.vol_learned(iid):
            return 1
        return 0


class DataLog:
    FILES = {
        "books": ["t", "iid", "bid", "bid_vol", "ask", "ask_vol", "fair", "theo"],
        "ticks": ["t", "iid", "price", "volume", "aggressor", "buyer", "seller"],
        "fills": ["t", "iid", "side", "price", "volume", "fair", "edge_ticks", "aggressive", "arm"],
        "news": ["t", "post", "meta"],
    }

    def __init__(self, data_dir=C.DATA_DIR):
        self.dir = os.path.join(data_dir, "raw", time.strftime("%Y%m%d_%H%M%S"))
        self.data_dir = data_dir
        os.makedirs(self.dir, exist_ok=True)
        self.fh, self.w = {}, {}
        for name, header in self.FILES.items():
            fh = open(os.path.join(self.dir, name + ".csv"), "a", newline="", encoding="utf-8")
            self.fh[name] = fh
            self.w[name] = csv.writer(fh)
            self.w[name].writerow(header)

    def row(self, name, values):
        self.w[name].writerow(values)

    def flush(self):
        for fh in self.fh.values():
            fh.flush()

    def close(self):
        for fh in self.fh.values():
            fh.close()

    def session_summary(self, row):
        path = os.path.join(self.data_dir, "sessions.csv")
        new = not os.path.exists(path)
        with open(path, "a", newline="") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(["start", "minutes", "session_pnl", "lots", "notional", "pnl_per_min"])
            w.writerow(row)
