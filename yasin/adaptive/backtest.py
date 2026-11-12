"""
Replay backtest on the 4-hour market recording (books.pkl + trades.pkl from the
other-teams analysis). Runs the real bot code against history, offline.

Fill model (per historical aggressive order = trades with the same time/stock/side):
  our resting ask at price a gets min(our volume, volume that order traded at prices
  WORSE than a) -- i.e. the taker would have hit us first. Same for bids. Our IOCs fill
  against the recorded best bid/ask. Our orders don't move the market (no feedback),
  and at equal price we assume we're behind the queue (no fill).
Takers are the real historical ones, so informed flow and jumps are in the test.

    python backtest.py <folder with books.pkl/trades.pkl> [--quick]
"""
import math
import os
import shutil
import sys
import tempfile
import time as _time
import types
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config as C      # noqa: E402

NS = types.SimpleNamespace
STEP_S = 0.25


def load(folder, t_from=None, t_to=None):
    import pandas as pd
    books = pd.read_pickle(os.path.join(folder, "books.pkl"))
    trades = pd.read_pickle(os.path.join(folder, "trades.pkl"))
    books = books.sort_values("timestamp_ns")
    trades = trades.sort_values("timestamp_ns")
    if t_from is not None:
        books = books[books.timestamp_ns >= t_from]
        trades = trades[trades.timestamp_ns >= t_from]
    if t_to is not None:
        books = books[books.timestamp_ns < t_to]
        trades = trades[trades.timestamp_ns < t_to]
    cols = ["timestamp_ns", "instrument_id", "bid", "ask", "wall_bid", "wall_bid_vol",
            "wall_ask", "wall_ask_vol", "in_bid_vol", "in_ask_vol"]
    brows = list(books[cols].itertuples(index=False, name=None))
    trows = list(trades[["timestamp_ns", "instrument_id", "price", "volume", "aggressor_side"]]
                 .itertuples(index=False, name=None))
    return brows, trows


def _book_from_row(r):
    _, _, bid, ask, wb, wbv, wa, wav, ibv, iav = r
    bids, asks = [], []
    if not math.isnan(bid):
        if not math.isnan(wb) and bid > wb + 1e-9:
            bids.append(NS(price=bid, volume=int(max(1, ibv or 1))))
        elif math.isnan(wb):
            bids.append(NS(price=bid, volume=int(max(1, ibv or 1))))
    if not math.isnan(wb):
        bids.append(NS(price=wb, volume=int(wbv)))
    if not math.isnan(ask):
        if not math.isnan(wa) and ask < wa - 1e-9:
            asks.append(NS(price=ask, volume=int(max(1, iav or 1))))
        elif math.isnan(wa):
            asks.append(NS(price=ask, volume=int(max(1, iav or 1))))
    if not math.isnan(wa):
        asks.append(NS(price=wa, volume=int(wav)))
    return NS(bids=bids, asks=asks)


class BacktestExchange:
    def __init__(self, brows, trows, tick=0.1):
        self.brows, self.trows = brows, trows
        self.bi = self.ti = 0
        ids = sorted({r[1] for r in brows})
        self.inst = {i: NS(instrument_id=i, tick_size=tick, instrument_type=None, instrument_group=None,
                           base_instrument_id=None, interest_rate=None, expiry=None, option_kind=None,
                           strike=None, index_id=None, index_constituents=None, index_divisor=None,
                           index_volatility=None, etf_cash_comp=None, etf_multiplier=None, price_limit=None)
                     for i in ids}
        self.latest = {}
        self.wall_mid = {}
        self.ticks = defaultdict(list)
        self.my_trades = defaultdict(list)
        self.orders = {i: {} for i in ids}
        self.pos = defaultdict(int)
        self.cash = 0.0
        self.next_id = 1
        self.requests = 0
        self.fills_passive = self.fills_ioc = 0
        self.fill_log = []                          # (t_s, iid, sgn, price, vol, kind)
        self.wm_hist = defaultdict(list)            # iid -> [(t_s, wall_mid)]
        self.now_s = 0.0

    # ---- replay
    def advance_to(self, t_ns):
        while self.bi < len(self.brows) and self.brows[self.bi][0] <= t_ns:
            r = self.brows[self.bi]
            self.latest[r[1]] = _book_from_row(r)
            if not math.isnan(r[4]) and not math.isnan(r[6]):
                self.wall_mid[r[1]] = 0.5 * (r[4] + r[6])
                self.wm_hist[r[1]].append((r[0] / 1e9, self.wall_mid[r[1]]))
            self.bi += 1
        groups = defaultdict(list)                  # one aggressive order = same time/stock/side
        while self.ti < len(self.trows) and self.trows[self.ti][0] <= t_ns:
            ts, iid, p, v, side = self.trows[self.ti]
            groups[(ts, iid, side)].append((p, int(v)))
            self.ticks[iid].append(NS(price=p, volume=int(v), aggressor_side=side, buyer="", seller=""))
            self.ti += 1
        for (ts, iid, side), prints in groups.items():
            self._match(iid, side, prints)

    def _match(self, iid, aggr_side, prints):
        rest_side = "ask" if aggr_side == "bid" else "bid"
        mine = sorted((o for o in self.orders.get(iid, {}).values() if o.side == rest_side),
                      key=lambda o: o.price if rest_side == "ask" else -o.price)
        for o in mine:
            if rest_side == "ask":
                worse = sum(v for p, v in prints if p > o.price + 1e-9)
            else:
                worse = sum(v for p, v in prints if p < o.price - 1e-9)
            q = min(o.volume, worse)
            if q <= 0:
                continue
            self._fill(iid, rest_side, o.price, q, o.order_id, "passive")
            self.fills_passive += q
            o.volume -= q
            if o.volume <= 0:
                del self.orders[iid][o.order_id]
            # consume that taker volume so it isn't given twice
            left = q
            new = []
            for p, v in sorted(prints, key=lambda x: -x[0] if rest_side == "ask" else x[0]):
                take = min(v, left) if ((p > o.price) if rest_side == "ask" else (p < o.price)) else 0
                left -= take
                if v - take > 0:
                    new.append((p, v - take))
            prints = new

    def _fill(self, iid, side, price, vol, oid, kind="passive"):
        sgn = 1 if side == "bid" else -1
        self.fill_log.append((self.now_s, iid, sgn, price, vol, kind))
        self.pos[iid] += sgn * vol
        self.cash -= sgn * price * vol
        self.my_trades[iid].append(NS(volume=vol, price=price, side=side, order_id=oid))

    # ---- API used by the bot
    def is_connected(self): return True
    def get_tradable_instruments(self): return dict(self.inst)
    def get_last_price_book(self, iid): return self.latest.get(iid)
    def get_positions(self): return {i: self.pos[i] for i in self.inst}
    def get_outstanding_orders(self, iid): return dict(self.orders[iid])

    def poll_new_trades(self, iid):
        t, self.my_trades[iid] = self.my_trades[iid], []
        return t

    def poll_new_trade_ticks(self, iid):
        t, self.ticks[iid] = self.ticks[iid], []
        return t

    def delete_order(self, iid, order_id):
        self.requests += 1
        self.orders[iid].pop(order_id, None)
        return NS(success=True)

    def delete_orders(self, iid):
        self.requests += 1
        self.orders[iid].clear()

    def insert_order(self, iid, price, volume, side, order_type):
        self.requests += 1
        assert abs(round(price / 0.1) * 0.1 - price) < 1e-6, f"off-tick {iid} {price}"
        new_pos = self.pos[iid] + (volume if side == "bid" else -volume)
        if abs(new_pos) > C.POSITION_LIMIT:
            return NS(success=False, order_id=None, error_reason="position limit")
        oid = self.next_id
        self.next_id += 1
        book = self.latest.get(iid)
        if order_type == "ioc":
            lvl = (book.asks[0] if side == "bid" else book.bids[0]) if book else None
            if lvl and ((price >= lvl.price) if side == "bid" else (price <= lvl.price)):
                q = min(volume, lvl.volume)
                self._fill(iid, side, lvl.price, q, oid, "ioc")
                self.fills_ioc += q
            return NS(success=True, order_id=oid, error_reason=None)
        self.orders[iid][oid] = NS(order_id=oid, price=price, volume=int(volume), side=side)
        return NS(success=True, order_id=oid, error_reason=None)

    def mark(self):
        return self.cash + sum(p * self.wall_mid.get(i, 0.0) for i, p in self.pos.items())

    def get_pnl(self, valuations=None):
        return self.mark()


def _wm_at(hist, t):
    import bisect
    i = bisect.bisect_right(hist, (t, float("inf"))) - 1
    return hist[i][1] if i >= 0 else None


def breakdown(ex):
    """Per (kind, stock): lots, edge at fill vs wall mid, value 5 s / 30 s later (cash units)."""
    agg = defaultdict(lambda: [0, 0.0, 0.0, 0.0])
    for t, iid, sgn, price, vol, kind in ex.fill_log:
        h = ex.wm_hist.get(iid)
        if not h:
            continue
        w0, w5, w30 = _wm_at(h, t), _wm_at(h, t + 5), _wm_at(h, t + 30)
        if None in (w0, w5, w30):
            continue
        for key in ((kind, iid), (kind, "ALL")):
            a = agg[key]
            a[0] += vol
            a[1] += sgn * (w0 - price) * vol
            a[2] += sgn * (w5 - price) * vol
            a[3] += sgn * (w30 - price) * vol
    return dict(agg)


PROFILES = {
    # what ran live in sessions 4/5: microprice fair, vol-based edge, no protection
    "old": dict(USE_WALLS=False, QUOTE_INSIDE=False, JUMP_GUARD=False, PAIRS=[], QUOTE_SIZES={},
                QUOTE_VOLUME=5, MAX_POS=60, FILL_COOLDOWN_S=0.0, CLOCK_AWARE=False, LEADS={},
                LOOP_SLEEP=0.25, SKEW_OWN=0.5, SKEW_ROOT=0.7, SKEW_CONVEXITY=2.0),
    "new": dict(),
    "no_clock": dict(CLOCK_AWARE=False),
    "no_leadlag": dict(LEADS={}),
    "clock_wide": dict(CLOCK_WIDE=True),
    "no_pairs": dict(PAIRS=[]),
    "caps20_30": dict(SOFT_CAP_NARROW=20, SOFT_CAP_WIDE=30),
    "size10": dict(QUOTE_VOLUME=10),
    # closest to the teammate's live settings: no pair netting, passive off on AVGO/AMD/NVDA, their betas
    "teammate_like": dict(PAIRS=[], QUOTE_SIZES={"AVGO": 0, "AMD": 0, "NVDA": 0},
                          LEADS={"NBIS": ("AVGO", 0.22), "CRWV": ("MU", 0.95), "SNDK": ("NVDA", 0.40)}),
}


def run(folder, profile, brows, trows, verbose=False):
    saved = {k: getattr(C, k) for k in dir(C) if k.isupper()}
    for k, v in PROFILES[profile].items():
        setattr(C, k, v)
    C.AUTO_PRETRAIN = False
    import bot as B          # reads config values at call time, so no reload needed
    clock = [brows[0][0] / 1e9]
    B.time = NS(time=lambda: clock[0], strftime=_time.strftime, localtime=_time.localtime,
                sleep=lambda s: clock.__setitem__(0, clock[0] + s))
    if not verbose:
        B.print = lambda *a, **k: None
    tmp = tempfile.mkdtemp(prefix="bt_")
    ex = BacktestExchange(brows, trows)
    curve = []
    max_abs = 0
    try:
        ex.advance_to(brows[0][0])
        bot = B.AdaptiveBot(ex, data_dir=tmp)
        end = brows[-1][0] / 1e9
        next_mark = clock[0]
        step_s = C.LOOP_SLEEP
        while clock[0] < end:
            ex.now_s = clock[0]
            ex.advance_to(int(clock[0] * 1e9))
            bot.step()
            max_abs = max([max_abs] + [abs(p) for p in ex.pos.values()])
            if clock[0] >= next_mark:
                curve.append((clock[0], ex.mark()))
                next_mark += 60
            clock[0] += step_s
        bot.log.close()
    finally:
        for k, v in saved.items():
            setattr(C, k, v)
        shutil.rmtree(tmp, ignore_errors=True)
    pnl = ex.mark()
    peak, dd = -1e18, 0.0
    for _, v in curve:
        peak = max(peak, v)
        dd = min(dd, v - peak)
    return dict(profile=profile, pnl=pnl, breakdown=breakdown(ex), clock_pulls=getattr(bot, "blackouts", 0),
                phases={k: round(v, 3) for k, v in getattr(bot, "phase", {}).items()}, passive=ex.fills_passive, ioc=ex.fills_ioc,
                max_pos=max_abs, max_dd=dd, jumps=getattr(bot, "jumps", 0),
                req_per_s=ex.requests / max(1.0, end - brows[0][0] / 1e9),
                hours=(end - brows[0][0] / 1e9) / 3600, curve=curve,
                by_stock={i: round(ex.pos[i], 0) for i in ex.inst})


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else \
        os.path.join(HERE, "..", "..", "..", "research run")
    profiles = [a.split("=", 1)[1] for a in sys.argv if a.startswith("--profile=")] or list(PROFILES)
    t_from = t_to = None
    for a in sys.argv:
        if a.startswith("--from="):
            t_from = int(a.split("=", 1)[1])
        if a.startswith("--to="):
            t_to = int(a.split("=", 1)[1])
    brows, trows = load(folder, t_from, t_to)
    print(f"loaded {len(brows)} books, {len(trows)} trades")
    for prof in profiles:
        t0 = _time.time()
        r = run(folder, prof, brows, trows)
        print(f"{prof:<14} PnL {r['pnl']:9.2f} | {r['hours']:.2f} h -> {r['pnl'] / r['hours']:8.2f}/h | "
              f"passive lots {r['passive']:6d} ioc lots {r['ioc']:5d} | max|pos| {r['max_pos']:3d} | "
              f"max drawdown {r['max_dd']:8.2f} | jumps {r['jumps']:4d} | clock pulls {r['clock_pulls']:5d} | req/s {r['req_per_s']:.1f} | "
              f"{_time.time() - t0:.0f}s", flush=True)
        if "--detail" in sys.argv:
            for (kind, iid), (lots, e0, e5, e30) in sorted(r["breakdown"].items()):
                print(f"    {kind:<8}{iid:<6} lots {lots:6d} | per lot (ticks): at fill {10 * e0 / lots:+6.2f}  "
                      f"5s {10 * e5 / lots:+6.2f}  30s {10 * e30 / lots:+6.2f} | 30s value {e30:+8.1f}")
