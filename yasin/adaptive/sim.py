"""
Fake exchange to smoke-test the adaptive bot over several sessions (not a real
backtest). Runs sessions back to back with one shared brain, so you can watch
tiers unlock and see whether session PnL improves.

    python yasin/adaptive/sim.py [sessions] [steps_per_session]
"""
import enum
import math
import os
import random
import shutil
import sys
import tempfile
import time as _time
import types
from datetime import datetime, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bot as B                                     # noqa: E402
import config as C                                  # noqa: E402
from pricing import bs_price                        # noqa: E402

NS = types.SimpleNamespace
TICK = 0.1
SIGMA_STEP = 0.03          # stdev of the true price per step (0.25s)
YEAR_S = 365 * 24 * 3600
TRUE_VOL = SIGMA_STEP / 100 / math.sqrt(C.LOOP_SLEEP / YEAR_S)


class IT(enum.Enum):
    STOCK = 1; STOCK_OPTION = 2; STOCK_FUTURE = 3; INDEX_TRACKING_ETF = 4; INDEX_OPTION = 5; INDEX_FUTURE = 6  # noqa: E702


class OK(enum.Enum):
    CALL = 1; PUT = 2  # noqa: E702


def inst(iid, typ, **kw):
    d = dict(instrument_id=iid, tick_size=TICK, instrument_type=typ, instrument_group=None,
             base_instrument_id=None, interest_rate=None, expiry=None, option_kind=None, strike=None,
             index_id=None, index_constituents=None, index_divisor=None, index_volatility=None,
             etf_cash_comp=None, etf_multiplier=None, price_limit=None)
    d.update(kw)
    return NS(**d)


class FakeExchange:
    def __init__(self, seed, state=None):
        self.rng = random.Random(seed)
        exp_f = datetime.now() + timedelta(days=60)
        exp_o = datetime.now() + timedelta(days=1)
        self.inst = {
            "A": inst("A", IT.STOCK, instrument_group="X"),
            "B": inst("B", IT.STOCK, instrument_group="X"),
            "F": inst("F", IT.STOCK_FUTURE, base_instrument_id="A", interest_rate=0.03, expiry=exp_f),
            "C100": inst("C100", IT.STOCK_OPTION, base_instrument_id="A", option_kind=OK.CALL,
                         strike=100.0, expiry=exp_o, index_volatility=TRUE_VOL),
            "P": inst("P", IT.STOCK),
            "Q": inst("Q", IT.STOCK),
            "E": inst("E", IT.INDEX_TRACKING_ETF, index_id=1.0, index_constituents={"P": 1.0, "Q": 2.0},
                      index_divisor=3.0, etf_multiplier=1.0, etf_cash_comp=0.0),
        }
        st = state or {}
        self.px = st.get("px", {"A": 100.0, "P": 50.0, "Q": 80.0})
        self.pos = st.get("pos", {k: 0 for k in self.inst})
        self.cash = st.get("cash", 0.0)
        self.orders = {k: {} for k in self.inst}
        self.next_id = 1
        self.trades = {k: [] for k in self.inst}
        self.ticks = {k: [] for k in self.inst}
        self.news = []
        self.books = {}
        self.requests = 0
        self.next_move = {k: self.rng.gauss(0, SIGMA_STEP) for k in self.px}
        self.advance(first=True)

    def state(self):
        return {"px": dict(self.px), "pos": dict(self.pos), "cash": self.cash}

    def value(self, iid):
        if iid in ("A", "B"):
            return self.px["A"]
        if iid == "F":
            T = (self.inst["F"].expiry - datetime.now()).total_seconds() / YEAR_S
            return self.px["A"] * math.exp(0.03 * T)
        if iid == "C100":
            T = (self.inst["C100"].expiry - datetime.now()).total_seconds() / YEAR_S
            return bs_price(True, self.px["A"], 100.0, T, 0.0, TRUE_VOL)
        if iid == "E":
            return (self.px["P"] + 2 * self.px["Q"]) / 3.0
        return self.px[iid]

    def advance(self, first=False):
        if not first:
            for k in self.px:
                self.px[k] += self.next_move[k]
                self.next_move[k] = self.rng.gauss(0, SIGMA_STEP)
            if self.rng.random() < 0.003:                       # news: a jump next step
                k = self.rng.choice(list(self.px))
                self.next_move[k] += self.rng.choice([-1, 1]) * 1.0
                self.news.append(NS(post=f"Big news about {k}!", meta_data=None))
        for iid in self.inst:
            v = self.value(iid)
            if iid == "B" and self.rng.random() < 0.05:
                v += self.rng.choice([-1, 1]) * 0.5
            half = self.rng.choice([0.1, 0.2, 0.3])
            bid = round(math.floor((v - half) / TICK) * TICK, 6)
            ask = round(math.ceil((v + half) / TICK) * TICK, 6)
            if ask - bid < TICK - 1e-9:
                ask = round(bid + TICK, 6)
            self.books[iid] = NS(bids=[NS(price=max(TICK, bid), volume=self.rng.randint(5, 30))],
                                 asks=[NS(price=max(2 * TICK, ask), volume=self.rng.randint(5, 30))])
        if first:
            return
        for iid in self.inst:
            # an informed trader ("SHARK") trades in the direction of the next move
            root = "A" if iid in ("A", "B", "F", "C100") else ("P" if iid == "E" else iid)
            nm = self.next_move[root] if root in self.next_move else 0.0
            if self.rng.random() < 0.3 and abs(nm) > SIGMA_STEP * 0.5:
                side = "bid" if nm > 0 else "ask"
                self.ticks[iid].append(NS(volume=self.rng.randint(1, 10), price=self.value(iid), aggressor_side=side,
                                          buyer="SHARK" if side == "bid" else "noise1",
                                          seller="SHARK" if side == "ask" else "noise1"))
            side = self.rng.choice(["bid", "ask"])
            nm_name = self.rng.choice(["noise1", "noise2", "noise3"])
            self.ticks[iid].append(NS(volume=self.rng.randint(1, 20), price=self.value(iid), aggressor_side=side,
                                      buyer=nm_name if side == "bid" else "noise4",
                                      seller=nm_name if side == "ask" else "noise4"))
        # takers on our resting orders: informed (order is wrong vs true value) or noise
        for iid, ords in self.orders.items():
            v = self.value(iid)
            for oid, o in list(ords.items()):
                informed = (o.side == "ask" and o.price < v) or (o.side == "bid" and o.price > v)
                if informed or self.rng.random() < 0.15:
                    q = min(o.volume, self.rng.randint(1, 5))
                    self._fill(iid, o.side, o.price, q, oid)
                    o.volume -= q
                    if o.volume <= 0:
                        del ords[oid]

    def _fill(self, iid, side, price, vol, oid):
        sgn = 1 if side == "bid" else -1
        self.pos[iid] += sgn * vol
        self.cash -= sgn * price * vol
        self.trades[iid].append(NS(volume=vol, price=price, side=side, order_id=oid))

    # ---- API ----
    def is_connected(self): return True
    def get_tradable_instruments(self): return dict(self.inst)
    def get_last_price_book(self, iid): return self.books[iid]
    def get_positions(self): return dict(self.pos)
    def get_outstanding_orders(self, iid): return dict(self.orders[iid])

    def poll_new_trades(self, iid):
        t, self.trades[iid] = self.trades[iid], []
        return t

    def poll_new_trade_ticks(self, iid):
        t, self.ticks[iid] = self.ticks[iid], []
        return t

    def poll_new_social_media_feeds(self):
        n, self.news = self.news, []
        return n

    def delete_order(self, iid, order_id):
        self.requests += 1
        self.orders[iid].pop(order_id, None)

    def delete_orders(self, iid):
        self.requests += 1
        self.orders[iid].clear()

    def insert_order(self, iid, price, volume, side, order_type):
        self.requests += 1
        assert abs(round(price / TICK) * TICK - price) < 1e-6, f"off-tick {iid} {price}"
        assert isinstance(volume, int) and volume >= 1, f"bad volume {volume}"
        assert price > 0, f"non-positive price {iid} {price}"
        new_pos = self.pos[iid] + (volume if side == "bid" else -volume)
        if abs(new_pos) > C.POSITION_LIMIT:
            return NS(success=False, order_id=None, error_reason="position limit")
        oid = self.next_id
        self.next_id += 1
        b = self.books[iid]
        if order_type == "ioc":
            lvl = b.asks[0] if side == "bid" else b.bids[0]
            if (price >= lvl.price) if side == "bid" else (price <= lvl.price):
                q = min(volume, lvl.volume)
                if q > 0:
                    self._fill(iid, side, lvl.price, q, oid)
                    lvl.volume -= q
            return NS(success=True, order_id=oid, error_reason=None)
        crosses = price >= b.asks[0].price if side == "bid" else price <= b.bids[0].price
        assert not crosses, f"limit order crosses book: {iid} {side} {price}"
        self.orders[iid][oid] = NS(order_id=oid, price=price, volume=volume, side=side)
        return NS(success=True, order_id=oid, error_reason=None)

    def get_pnl(self, valuations=None):
        return self.cash + sum(p * self.value(i) for i, p in self.pos.items())


clock = [1_000_000.0]
B.time = NS(time=lambda: clock[0], strftime=_time.strftime, localtime=_time.localtime,
            sleep=lambda s: clock.__setitem__(0, clock[0] + s))


def run_session(seed, steps, data_dir, state, quiet=True, show_report=False):
    ex = FakeExchange(seed, state)
    bot = B.AdaptiveBot(ex, data_dir=data_dir)
    pnl0 = ex.get_pnl()
    bot.start_pnl = pnl0
    max_abs = 0
    real_print = print
    if quiet:
        B.print = lambda *a, **k: None
    try:
        for i in range(steps):
            bot.step()
            clock[0] += C.LOOP_SLEEP
            ex.advance()
            max_abs = max(max_abs, max(abs(p) for p in ex.pos.values()))
    finally:
        B.print = real_print
    if show_report:
        print("\nLast session report:")
    else:
        B.print = lambda *a, **k: None
    bot.report()
    B.print = real_print
    pnl = ex.get_pnl() - pnl0
    bot.shutdown()
    return bot, ex, pnl, max_abs


def main():
    sessions = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 2400
    data_dir = tempfile.mkdtemp(prefix="adaptive_sim_")
    print(f"true annual vol {TRUE_VOL:.2f}, sim data in {data_dir}")
    state = None
    try:
        for s in range(sessions):
            bot, ex, pnl, mx = run_session(s, steps, data_dir, state, show_report=(s == sessions - 1))
            state = ex.state()
            b = bot.brain
            tiers = {i: b.tier(i) for i in bot.instruments}
            alpha = {k: ("ON" if b.alpha_active(k) else "off", r["n"]) for k, r in b.s["reg"].items()}
            shark = b.s["names"].get("SHARK", {})
            print(f"session {s + 1}: PnL {pnl:9.2f} | lots {bot.lots:5d} | max|pos| {mx:3d} | "
                  f"req/step {ex.requests / steps:.2f} | tiers {tiers}")
            print(f"           alpha {alpha} | SHARK score {b.name_score('SHARK'):+.2f} (n={shark.get('n')}) | "
                  f"news pause {b.news_pause():.0f}s | arms "
                  f"{ {i: max(a, key=lambda k: a[k]['mean']) for i, a in b.s['bandit'].items() if len(a) > 1} }")
    finally:
        shutil.rmtree(data_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
