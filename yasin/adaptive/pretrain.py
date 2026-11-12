"""
Pre-train the brain from historical data (the "context") -- as a PRIOR, not as live data.

Replays raw_pricebooks.csv.gz + raw_trades.csv.gz (full depth, so the exchange walls
are visible) through the bot's own market-data / fair-value / learning code into a
THROWAWAY brain, then hands the real brain only:
  - per-stock volatility as a prior (worth HIST_PRIOR_VOL_SAMPLES live samples, and it
    drops out by itself when live volatility disagrees), and
  - prediction-model samples at HIST_SAMPLE_WEIGHT each. Whether that model may act is
    decided by its accuracy on LIVE data only.
Nothing here counts as live evidence: no tier opens because of history.
No exchange connection, no orders. Each history file is used once (remembered in brain.json).

    python pretrain.py            # replay history the brain hasn't seen
    python pretrain.py --force    # replay again anyway
    python pretrain.py <folder>   # use a specific history folder
"""
import csv
import gzip
import json
import math
import os
import shutil
import sys
import tempfile
import time as _time
import types
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C          # noqa: E402

NS = types.SimpleNamespace
BOOKS, TRADES = "raw_pricebooks.csv.gz", "raw_trades.csv.gz"


def find_history_dir(explicit=None):
    for d in ([explicit] if explicit else C.HISTORY_DIRS):
        if d and os.path.exists(os.path.join(d, BOOKS)):
            return os.path.abspath(d)
    return None


def _signature(path):
    st = os.stat(path)
    return f"{os.path.basename(os.path.dirname(path))}/{os.path.basename(path)}:{st.st_size}"


def load_history(folder):
    """Full-depth book snapshots (t, iid, bids, asks) with levels as (price, volume),
    and trades, both sorted by time (seconds)."""
    books = []
    with gzip.open(os.path.join(folder, BOOKS), "rt", newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                bids = [(float(p), int(v)) for p, v in json.loads(row["bids_json"])]
                asks = [(float(p), int(v)) for p, v in json.loads(row["asks_json"])]
            except (ValueError, KeyError, TypeError):
                continue
            if not bids or not asks or asks[0][0] <= bids[0][0]:
                continue
            books.append((int(row["timestamp_ns"]) / 1e9, row["instrument_id"], bids, asks))
    trades = []
    with gzip.open(os.path.join(folder, TRADES), "rt", newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                trades.append((int(row["timestamp_ns"]) / 1e9, row["instrument_id"], float(row["price"]),
                               int(float(row["volume"])), row.get("aggressor_side") or None,
                               row.get("buyer") or "", row.get("seller") or ""))
            except (ValueError, KeyError):
                continue
    books.sort(key=lambda r: r[0])
    trades.sort(key=lambda r: r[0])
    return books, trades


def infer_tick(prices):
    """Smallest price step seen, snapped to a 1/2/5 x 10^k grid."""
    ps = sorted({round(p, 6) for p in prices})
    diffs = [b - a for a, b in zip(ps, ps[1:]) if b - a > 1e-6]
    if not diffs:
        return 0.01
    d = min(diffs)
    k = 10 ** math.floor(math.log10(d))
    for m in (1, 2, 5, 10):
        if d <= m * k * 1.05:
            return round(m * k, 10)
    return round(d, 10)


class ReplayExchange:
    """Serves historical books/trades as if live. Never fills anything."""

    def __init__(self, books, trades, ticks):
        self.books, self.trades = books, trades
        self.bi = self.ti = 0
        self.latest = {}
        self.queue = defaultdict(list)
        self.inst = {iid: NS(instrument_id=iid, tick_size=t, instrument_type=None, instrument_group=None,
                             base_instrument_id=None, interest_rate=None, expiry=None, option_kind=None,
                             strike=None, index_id=None, index_constituents=None, index_divisor=None,
                             index_volatility=None, etf_cash_comp=None, etf_multiplier=None, price_limit=None)
                     for iid, t in ticks.items()}

    def advance_to(self, t):
        while self.bi < len(self.books) and self.books[self.bi][0] <= t:
            _, iid, bids, asks = self.books[self.bi]
            self.latest[iid] = NS(bids=[NS(price=p, volume=v) for p, v in bids],
                                  asks=[NS(price=p, volume=v) for p, v in asks])
            self.bi += 1
        while self.ti < len(self.trades) and self.trades[self.ti][0] <= t:
            _, iid, p, v, side, buyer, seller = self.trades[self.ti]
            self.queue[iid].append(NS(price=p, volume=v, aggressor_side=side, buyer=buyer, seller=seller))
            self.ti += 1

    def get_tradable_instruments(self): return dict(self.inst)
    def get_last_price_book(self, iid): return self.latest.get(iid)

    def poll_new_trade_ticks(self, iid):
        q, self.queue[iid] = self.queue[iid], []
        return q


def pretrain(data_dir=C.DATA_DIR, history_dir=None, force=False, verbose=True):
    say = print if verbose else (lambda *a, **k: None)
    folder = find_history_dir(history_dir)
    if folder is None:
        say("Pretrain: no history folder found (looked in config.HISTORY_DIRS); skipping.")
        return False

    import bot as B
    from brain import FEATURE_NAMES, Brain

    brain = Brain(data_dir)
    sigs = [_signature(os.path.join(folder, f)) for f in (BOOKS, TRADES)]
    done = brain.s.setdefault("pretrained", [])
    if not force and all(s in done for s in sigs):
        say(f"Pretrain: history in {folder} already learned; skipping.")
        return False

    t0 = _time.time()
    say(f"Pretrain: loading {folder} ...")
    books, trades = load_history(folder)
    if not books:
        say("Pretrain: no usable book snapshots; skipping.")
        return False
    prices = defaultdict(list)
    for _, iid, bids, asks in books:
        prices[iid] += (bids[0][0], asks[0][0])
    ticks = {iid: (C.HISTORY_TICK or infer_tick(p)) for iid, p in prices.items()}
    say(f"Pretrain: {len(books)} books, {len(trades)} trades, "
        f"{(books[-1][0] - books[0][0]) / 60:.1f} min, ticks {ticks}")

    ex = ReplayExchange(books, trades, ticks)
    clock = [books[0][0]]
    real_time = B.time
    B.time = NS(time=lambda: clock[0], strftime=_time.strftime, localtime=_time.localtime,
                sleep=lambda s: None)
    B.print = lambda *a, **k: None
    tmp = tempfile.mkdtemp(prefix="pretrain_")
    wall_seen, wall_spread = defaultdict(int), defaultdict(list)
    try:
        bot = B.AdaptiveBot(ex, data_dir=tmp)      # throwaway brain, logs and session counter
        bot._bandit_epochs = lambda now: None      # no fills in history: don't score arms
        prev = None
        end = books[-1][0]
        n_steps = 0
        while clock[0] <= end:
            ex.advance_to(clock[0])
            now = clock[0]
            dt = now - prev if prev is not None else 0.0
            prev = now
            bot.read_market(now, dt)
            bot.compute_fairs()
            bot.learn(now, dt)
            n_steps += 1
            for iid, w in bot.walls.items():
                if w:
                    wall_seen[iid] += 1
                    wall_spread[iid].append(round((w[1] - w[0]) / ticks[iid]))
            clock[0] += C.LOOP_SLEEP
        bot.log.close()
        hist = bot.brain
    finally:
        B.time = real_time
        del B.print                                # back to the builtin print
        shutil.rmtree(tmp, ignore_errors=True)

    # hand over: volatility as a prior, model samples down-weighted. Live counts untouched.
    for iid in ticks:
        h = hist.inst(iid)
        if h["sigma2"]:
            brain.set_hist_vol(iid, h["sigma2"], h["n_vol"])
    for k, hr in hist.s["reg"].items():
        r = brain._reg(k)
        wgt = C.HIST_SAMPLE_WEIGHT
        for i in range(len(r["Xty"])):
            r["Xty"][i] += wgt * hr["Xty"][i]
            for j in range(len(r["Xty"])):
                r["XtX"][i][j] += wgt * hr["XtX"][i][j]
        r["n_hist"] += hr["n_live"]
        r["n"] += hr["n"]
        brain._w_cache.pop(k, None)
    for sig in sigs:
        if sig not in done:
            done.append(sig)
    brain.save()

    say(f"Pretrain: done in {_time.time() - t0:.0f}s -> {brain.path}")
    for iid in sorted(ticks):
        sp = sorted(wall_spread[iid])
        med = sp[len(sp) // 2] if sp else None
        say(f"  {iid:<8} history vol1s={math.sqrt(brain.inst(iid)['sigma2_hist'] or 0):.4f} "
            f"(prior worth {C.HIST_PRIOR_VOL_SAMPLES} live samples) | walls seen "
            f"{100 * wall_seen[iid] / max(1, n_steps):.0f}% of the time, median wall spread {med} ticks")
    for k, hr in hist.s["reg"].items():
        w = [round(x, 3) for x in hist._weights(k)]
        ratio = hr["mse_model"] / hr["mse_zero"] if hr["mse_zero"] else float("nan")
        say(f"  prediction model {k}: {hr['n_live']} history samples (weight {C.HIST_SAMPLE_WEIGHT}), "
            f"history-only error vs 'no move' {ratio:.3f}; acts only once live data proves it "
            f"(now {100 * brain.alpha_scale(k):.0f}%)")
        say(f"    weights {dict(zip(FEATURE_NAMES, w))}")
    return True


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    pretrain(history_dir=args[0] if args else None, force="--force" in sys.argv)
