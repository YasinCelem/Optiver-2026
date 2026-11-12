"""Fake Optibook exchange to smoke-test yasin/simple_mm.py logic (not a real backtest)."""
import sys, types, enum, random, math
from datetime import datetime, timedelta

# ---- stub optibook modules ----
ob = types.ModuleType("optibook"); sc = types.ModuleType("optibook.synchronous_client"); ct = types.ModuleType("optibook.common_types")
class IT(enum.Enum):
    STOCK = 1; STOCK_OPTION = 2; STOCK_FUTURE = 3; INDEX_TRACKING_ETF = 4; INDEX_OPTION = 5; INDEX_FUTURE = 6
ct.InstrumentType = IT
sc.Exchange = object
sys.modules.update({"optibook": ob, "optibook.synchronous_client": sc, "optibook.common_types": ct})
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import simple_mm as m

NS = types.SimpleNamespace
TICK = 0.1

def inst(iid, typ, group=None, base=None, r=None, expiry=None):
    return NS(instrument_id=iid, tick_size=TICK, instrument_type=typ, instrument_group=group,
              base_instrument_id=base, interest_rate=r, expiry=expiry)

class FakeExchange:
    def __init__(self, seed):
        self.rng = random.Random(seed)
        self.inst = {
            "A": inst("A", IT.STOCK, "X"),
            "B": inst("B", IT.STOCK, "X"),
            "F": inst("F", IT.STOCK_FUTURE, base="A", r=0.03, expiry=datetime.now() + timedelta(days=60)),
        }
        self.true = 100.0
        self.pos = {k: 0 for k in self.inst}
        self.cash = 0.0
        self.orders = {k: {} for k in self.inst}
        self.next_id = 1
        self.trades = {k: [] for k in self.inst}
        self.ticks = {k: [] for k in self.inst}
        self.books = {}
        self.requests = 0
        self.advance()

    def value(self, iid):
        if iid == "F":
            return self.true * math.exp(0.03 * 60 / 365)
        return self.true

    def advance(self):
        self.true += self.rng.gauss(0, 0.08)
        for iid in self.inst:
            v = self.value(iid)
            if iid == "B" and self.rng.random() < 0.05:   # af en toe misprijzing -> arb-kans
                v += self.rng.choice([-1, 1]) * 0.5
            half = self.rng.choice([0.1, 0.2, 0.3])
            bid = round(math.floor((v - half) / TICK) * TICK, 6)
            ask = round(math.ceil((v + half) / TICK) * TICK, 6)
            self.books[iid] = NS(bids=[NS(price=bid, volume=self.rng.randint(5, 30))],
                                 asks=[NS(price=ask, volume=self.rng.randint(5, 30))])
        for iid in self.inst:   # handel tussen andere partijen
            self.ticks[iid].append(NS(volume=self.rng.randint(5, 40), price=self.value(iid)))
        # Takers: geïnformeerd (raken quotes die verkeerd staan t.o.v. true) + ruis
        for iid, ords in self.orders.items():
            v = self.value(iid)
            for oid, o in list(ords.items()):
                informed = (o.side == "ask" and o.price < v) or (o.side == "bid" and o.price > v)
                noise = self.rng.random() < 0.15
                if informed or noise:
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
        self.ticks[iid].append(NS(volume=vol, price=price))

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
    def delete_order(self, iid, order_id):
        self.requests += 1
        self.orders[iid].pop(order_id, None)
    def delete_orders(self, iid):
        self.requests += 1
        self.orders[iid].clear()
    def insert_order(self, iid, price, volume, side, order_type):
        self.requests += 1
        assert abs(round(price / TICK) * TICK - price) < 1e-6, f"off-tick {price}"
        assert isinstance(volume, int) and volume >= 1, f"bad volume {volume}"
        new_pos = self.pos[iid] + (volume if side == "bid" else -volume)
        if abs(new_pos) > m.POSITION_LIMIT:
            return NS(success=False, order_id=None, error_reason="position limit")
        b = self.books[iid]
        if order_type == "ioc":
            lvl = b.asks[0] if side == "bid" else b.bids[0]
            crosses = price >= lvl.price if side == "bid" else price <= lvl.price
            if crosses:
                q = min(volume, lvl.volume)
                self._fill(iid, side, lvl.price, q, -1)
                lvl.volume -= q
            return NS(success=True, order_id=-1, error_reason=None)
        # limit: we gaan ervan uit dat de bot nooit kruist
        crosses = price >= b.asks[0].price if side == "bid" else price <= b.bids[0].price
        assert not crosses, f"limit order crosses book: {iid} {side} {price}"
        oid = self.next_id; self.next_id += 1
        self.orders[iid][oid] = NS(order_id=oid, price=price, volume=volume, side=side)
        return NS(success=True, order_id=oid, error_reason=None)
    def get_pnl(self, valuations=None):
        val = valuations or {}
        return self.cash + sum(p * val.get(i, self.value(i)) for i, p in self.pos.items())

import time as _t
clock = [0.0]
m.time = NS(time=lambda: clock[0], strftime=_t.strftime, sleep=lambda s: None)

def run(seed, steps=2000, verbose=False):
    clock[0] = 1_000_000.0
    ex = FakeExchange(seed)
    bot = m.MarketMaker(ex)
    fair = {}
    max_abs = 0
    for _ in range(steps):
        fair = bot.step()
        clock[0] += m.LOOP_SLEEP
        if verbose and _ in (0, steps // 2):
            bot.report(fair)
        max_abs = max(max_abs, max(abs(p) for p in ex.pos.values()),
                      abs(bot.group_net("A", ex.pos)))
        ex.advance()
    if verbose:
        bot.report(fair)
    true_vals = {i: ex.value(i) for i in ex.inst}
    return ex.get_pnl(true_vals), ex.pos, max_abs, ex.requests / steps, dict(bot.analytics.our_volume)

if __name__ == "__main__":
    print("groups:", m.MarketMaker(FakeExchange(0)).groups)
    m.ANALYTICS_CSV = None
    run(0, verbose=True)
    for seed in range(5):
        pnl, pos, mx, rps, fills = run(seed)
        print(f"seed {seed}: PnL {pnl:8.2f}  eindpos {pos}  max|pos/net| {mx}  req/loop {rps:.2f}  fills {fills}")
