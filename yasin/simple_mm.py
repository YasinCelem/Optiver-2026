"""
Market maker v2 – "easy running" bot, met de zwaktes van v1 aangepakt.

Wat hij doet, per loop:
  1. FAIR VALUE
     - Microprice per boek: mid gewogen naar volume aan top-of-book
       (veel volume op de bid => prijs duwt omhoog).
     - Gekoppelde aandelen (dual listing, zelfde instrument_group) delen één
       fair value: liquiditeitsgewogen gemiddelde van hun microprices.
     - Futures: theo = fair(onderliggend) * exp(r * T).
  2. ARBITRAGE / TAKEN
     - Dual listing: bid van boek A > ask van boek B  => koop B, verkoop A (IOC).
     - Elk instrument: ask ver onder fair of bid ver boven fair => direct pakken.
  3. QUOTEN
     - Edge schaalt mee met volatiliteit (EMA van fair-bewegingen):
       rustig => smal (meer fills), wild => breed (niet geplukt worden).
     - Skew op eigen positie EN op netto groepspositie (aandeel + future +
       dual-listing tellen samen als één risico).
     - Hysterese: alleen herquoten als de prijs echt verschuift (spaart
       requests, minder kans op rate limit).
  4. RISICO
     - Limieten per instrument en per groep.
     - Noodrem op PnL gewaardeerd tegen ONZE fair value (niet last-traded).

Draaien op de Optibook-omgeving, vanuit de repo-root:
    python yasin/simple_mm.py
Stoppen: Ctrl+C (orders worden getrokken).
"""

import logging
import math
import time
from collections import defaultdict
from datetime import datetime

from optibook.synchronous_client import Exchange
from optibook.common_types import InstrumentType

logging.getLogger("client").setLevel("ERROR")

# ----------------------------------------------------------------------------
# Instellingen – pas aan op basis van de challenge-beschrijving
# ----------------------------------------------------------------------------
INSTRUMENTS = None          # None = alles behalve opties, of bv. ["NVDA", "NVDA_DUAL"]
SKIP_TYPES = {InstrumentType.STOCK_OPTION, InstrumentType.INDEX_OPTION}
LINKED_GROUPS = []          # handmatig koppelen als auto-detectie faalt, bv. [["PHILIPS_A", "PHILIPS_B"]]

# Positie / risico
POSITION_LIMIT = 100        # limiet van de exchange per instrument (check challenge!)
MAX_POS = 60                # onze eigen limiet per instrument
MAX_GROUP_NET = 60          # max netto positie over een gekoppelde groep
MAX_LOSS = -2000.0          # noodrem op PnL (gewaardeerd tegen fair value)

# Quoten
QUOTE_VOLUME = 5            # lots per quote
EDGE_MIN_TICKS = 1          # minimale afstand fair -> quote
EDGE_VOL_MULT = 1.5         # edge = max(EDGE_MIN_TICKS*tick, EDGE_VOL_MULT * sigma)
SKEW_OWN_TICKS = 3          # verschuiving bij eigen positie == MAX_POS
SKEW_GROUP_TICKS = 3        # verschuiving bij groepsnetto == MAX_GROUP_NET
REQUOTE_TOL_TICKS = 1       # bestaande order laten staan als hij binnen dit veilig ligt

# Taken / arbitrage
ARB_MIN_TICKS = 1           # min. winst per lot (in ticks) voor dual-listing arb
TAKE_MIN_TICKS = 3          # min. afstand tot fair voordat we een quote raken
TAKE_VOL_MULT = 3.0         # take-drempel = max(TAKE_MIN_TICKS*tick, TAKE_VOL_MULT * sigma)
MAX_TAKE_VOLUME = 10        # max lots per IOC

# Volatiliteit
VOL_HALFLIFE_LOOPS = 40     # EMA-halfwaardetijd in loops

# Loop
LOOP_SLEEP = 0.25           # seconden (houd >= 0.2 i.v.m. rate limits)
REPORT_EVERY = 10.0         # seconden tussen statusregels

# Analytics
MARKOUT_SECONDS = 5.0       # na hoeveel sec we checken of een fill goed was
ANALYTICS_CSV = "mm_analytics.csv"   # None = niet wegschrijven

EPS = 1e-9


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def round_down(price, tick):
    return round(math.floor(price / tick + EPS) * tick, 6)


def round_up(price, tick):
    return round(math.ceil(price / tick - EPS) * tick, 6)


def top_of_book(book):
    if not book or not book.bids or not book.asks:
        return None
    b, a = book.bids[0], book.asks[0]
    if a.price <= b.price:
        return None
    return b.price, b.volume, a.price, a.volume


def microprice(tob):
    bid, bvol, ask, avol = tob
    if bvol + avol <= 0:
        return (bid + ask) / 2.0
    # Veel volume op de bid -> meer kans dat de ask wordt geraakt -> prijs omhoog
    return (bid * avol + ask * bvol) / (bvol + avol)


def years_to_expiry(expiry):
    if expiry is None:
        return 0.0
    return max(0.0, (expiry - datetime.now()).total_seconds() / (365.0 * 24 * 3600))


class Throttle:
    """Print dezelfde melding hooguit eens per `every` seconden."""

    def __init__(self, every=5.0):
        self.every = every
        self.last = {}

    def __call__(self, key, msg):
        now = time.time()
        if now - self.last.get(key, 0.0) >= self.every:
            self.last[key] = now
            print(msg)


# ----------------------------------------------------------------------------
# Analytics
# ----------------------------------------------------------------------------
class Analytics:
    """
    Houdt bij hoe de bot presteert:
      - PnL nu, PnL sinds start, PnL per minuut
      - Edge bij fill: hoeveel beter dan fair value we handelden (ticks en bps)
      - Markout win-rate: % fills dat MARKOUT_SECONDS later nog winstgevend is
        t.o.v. de fair value (laag = we worden geplukt / adverse selection)
      - Rendement op omzet: sessie-PnL / verhandelde notional (bps)
      - Market share: ons volume / totaal marktvolume per instrument
      - Markt sinds start: % koersverandering per instrument (benchmark)
    """

    def __init__(self, instruments):
        self.instruments = instruments
        self.start_time = time.time()
        self.start_pnl = None
        self.start_fair = {}
        self.lots = 0
        self.notional = 0.0
        self.edge_cash = 0.0          # som van (fair - prijs) * sgn * volume op fill-moment
        self.edge_ticks = 0.0         # idem in ticks
        self.pending = []             # fills die nog op hun markout wachten
        self.markout_wins = 0
        self.markout_total = 0
        self.markout_cash = 0.0
        self.last_fair = {}
        self.our_volume = defaultdict(int)
        self.market_volume = defaultdict(int)
        self._csv_header_written = False

    def on_step(self, ex, fair):
        now = time.time()
        self.last_fair = dict(fair)
        for iid, inst in self.instruments.items():
            for t in ex.poll_new_trades(iid):
                sgn = 1 if t.side == "bid" else -1
                self.lots += t.volume
                self.notional += t.price * t.volume
                self.our_volume[iid] += t.volume
                f = fair.get(iid)
                if f is not None:
                    self.edge_cash += (f - t.price) * sgn * t.volume
                    self.edge_ticks += (f - t.price) * sgn * t.volume / inst.tick_size
                    self.pending.append((now, iid, sgn, t.price, t.volume))
            for tick in ex.poll_new_trade_ticks(iid):
                self.market_volume[iid] += tick.volume
            if iid in fair and iid not in self.start_fair:
                self.start_fair[iid] = fair[iid]

        # Markout: was de fill na X seconden nog goed?
        still = []
        for item in self.pending:
            ts, iid, sgn, price, vol = item
            if now - ts < MARKOUT_SECONDS or iid not in fair:
                still.append(item)
                continue
            pnl = (fair[iid] - price) * sgn * vol
            self.markout_total += vol
            self.markout_cash += pnl
            if pnl > 0:
                self.markout_wins += vol
        self.pending = still

    def report(self, pnl, positions, nets):
        elapsed = time.time() - self.start_time
        if pnl is not None and self.start_pnl is None:
            self.start_pnl = pnl
        session = (pnl - self.start_pnl) if (pnl is not None and self.start_pnl is not None) else None
        per_min = session / (elapsed / 60) if (session is not None and elapsed > 60) else None

        def pct(a, b):
            return 100.0 * a / b if b else None

        def fmt(x, spec="+.2f", suffix=""):
            return "n/a" if x is None else f"{x:{spec}}{suffix}"

        avg_edge_ticks = self.edge_ticks / self.lots if self.lots else None
        edge_bps = 1e4 * self.edge_cash / self.notional if self.notional else None
        win_rate = pct(self.markout_wins, self.markout_total)
        markout_bps = 1e4 * self.markout_cash / self.notional if self.notional else None
        ret_bps = 1e4 * session / self.notional if (session is not None and self.notional) else None

        share = {i: pct(self.our_volume[i], self.market_volume[i]) for i in self.instruments
                 if self.market_volume[i]}
        mkt = {}
        for iid, f0 in self.start_fair.items():
            f = self.last_fair.get(iid)
            if f is not None and f0:
                mkt[iid] = 100.0 * (f - f0) / f0

        h, rem = divmod(int(elapsed), 3600)
        print(f"\n=== {time.strftime('%H:%M:%S')} | running {h:02d}:{rem // 60:02d}:{rem % 60:02d} ===")
        print(f"PnL now: {fmt(pnl, '.2f')} | since start: {fmt(session)} | per min: {fmt(per_min)}")
        print(f"Fills: {self.lots} lots | notional {self.notional:,.0f} | return on turnover: {fmt(ret_bps, '+.1f', ' bps')}")
        print(f"Edge at fill: {fmt(avg_edge_ticks, '+.2f', ' ticks')} ({fmt(edge_bps, '+.1f', ' bps')}) | "
              f"markout {MARKOUT_SECONDS:.0f}s: win-rate {fmt(win_rate, '.0f', '%')}, {fmt(markout_bps, '+.1f', ' bps')}")
        print("Market share: " + (", ".join(f"{i} {v:.1f}%" for i, v in share.items()) or "n/a"))
        print("Market since start: " + (", ".join(f"{i} {v:+.2f}%" for i, v in mkt.items()) or "n/a"))
        print(f"Positions: {positions} | group net: {nets}")

        if ANALYTICS_CSV:
            self._write_csv([time.strftime('%Y-%m-%d %H:%M:%S'), round(elapsed), pnl, session, self.lots,
                             round(self.notional, 2), avg_edge_ticks, win_rate, ret_bps])
        return session

    def _write_csv(self, row):
        import csv
        try:
            with open(ANALYTICS_CSV, "a", newline="") as fh:
                w = csv.writer(fh)
                if not self._csv_header_written:
                    w.writerow(["time", "elapsed_s", "pnl", "session_pnl", "lots", "notional",
                                "edge_ticks", "markout_winrate_pct", "return_bps"])
                    self._csv_header_written = True
                w.writerow(row)
        except OSError as e:
            print(f"CSV schrijven faalde: {e}")


# ----------------------------------------------------------------------------
# Bot
# ----------------------------------------------------------------------------
class MarketMaker:
    def __init__(self, exchange):
        self.ex = exchange
        self.log = Throttle()
        self.instruments = self._select_instruments()
        self.group_of, self.groups = self._build_groups()
        self.sigma = {}                      # EMA van |delta fair| per instrument
        self.prev_fair = {}
        self.alpha = 1 - 0.5 ** (1.0 / VOL_HALFLIFE_LOOPS)
        self.analytics = Analytics(self.instruments)

    # -- setup ---------------------------------------------------------------
    def _select_instruments(self):
        tradable = self.ex.get_tradable_instruments()
        if INSTRUMENTS is not None:
            return {i: tradable[i] for i in INSTRUMENTS if i in tradable}
        return {i: inst for i, inst in tradable.items() if inst.instrument_type not in SKIP_TYPES}

    def _build_groups(self):
        """Groepeer instrumenten die hetzelfde onderliggende risico hebben."""
        group_of = {}
        # 1. handmatige koppelingen
        for n, grp in enumerate(LINKED_GROUPS):
            for iid in grp:
                if iid in self.instruments:
                    group_of[iid] = f"manual{n}"
        # 2. niet-futures: instrument_group of eigen id
        for iid, inst in self.instruments.items():
            if iid in group_of or self._is_future(inst):
                continue
            group_of[iid] = getattr(inst, "instrument_group", None) or iid
        # 3. futures: zelfde groep als hun onderliggende
        for iid, inst in self.instruments.items():
            if iid in group_of:
                continue
            base = inst.base_instrument_id
            group_of[iid] = group_of.get(base, base or iid)
        groups = defaultdict(list)
        for iid, g in group_of.items():
            groups[g].append(iid)
        return group_of, dict(groups)

    @staticmethod
    def _is_future(inst):
        return inst.instrument_type in (InstrumentType.STOCK_FUTURE, InstrumentType.INDEX_FUTURE)

    # -- fair value ----------------------------------------------------------
    def compute_fairs(self, tobs):
        fair = {}
        # Spot/gelijkwaardige legs: liquiditeitsgewogen microprice per groep
        for g, members in self.groups.items():
            num = den = 0.0
            for iid in members:
                if self._is_future(self.instruments[iid]) or iid not in tobs:
                    continue
                w = tobs[iid][1] + tobs[iid][3]
                num += microprice(tobs[iid]) * w
                den += w
            if den > 0:
                for iid in members:
                    if not self._is_future(self.instruments[iid]):
                        fair[iid] = num / den
        # Futures: theo uit onderliggende, anders eigen microprice
        for iid, inst in self.instruments.items():
            if not self._is_future(inst):
                if iid not in fair and iid in tobs:
                    fair[iid] = microprice(tobs[iid])
                continue
            base_fair = fair.get(inst.base_instrument_id)
            if base_fair is not None:
                r = inst.interest_rate or 0.0
                fair[iid] = base_fair * math.exp(r * years_to_expiry(inst.expiry))
            elif iid in tobs:
                fair[iid] = microprice(tobs[iid])
        return fair

    def update_vol(self, fair):
        for iid, f in fair.items():
            prev = self.prev_fair.get(iid)
            if prev is not None:
                move = abs(f - prev)
                s = self.sigma.get(iid, move)
                self.sigma[iid] = (1 - self.alpha) * s + self.alpha * move
            self.prev_fair[iid] = f

    # -- risico --------------------------------------------------------------
    def group_net(self, iid, positions):
        return sum(positions.get(m, 0) for m in self.groups[self.group_of[iid]])

    def buy_room(self, iid, positions):
        pos = positions.get(iid, 0)
        return max(0, min(MAX_POS - pos, POSITION_LIMIT - pos, MAX_GROUP_NET - self.group_net(iid, positions)))

    def sell_room(self, iid, positions):
        pos = positions.get(iid, 0)
        return max(0, min(MAX_POS + pos, POSITION_LIMIT + pos, MAX_GROUP_NET + self.group_net(iid, positions)))

    def ioc(self, iid, side, price, volume):
        if volume <= 0:
            return False
        resp = self.ex.insert_order(iid, price=price, volume=int(volume), side=side, order_type="ioc")
        if not resp.success:
            self.log(f"ioc-{iid}", f"[{iid}] IOC {side} {volume}@{price} afgewezen: {resp.error_reason}")
        return resp.success

    # -- stap 2: arbitrage + taken -------------------------------------------
    def arbitrage(self, tobs, positions):
        """Dual listing: verkoop het dure boek, koop het goedkope. Return True als er gehandeld is."""
        traded = False
        for members in self.groups.values():
            legs = [m for m in members if m in tobs and not self._is_future(self.instruments[m])]
            for a in legs:          # verkopen in a
                for b in legs:      # kopen in b
                    if a == b:
                        continue
                    bid_a, bvol_a, _, _ = tobs[a]
                    _, _, ask_b, avol_b = tobs[b]
                    tick = max(self.instruments[a].tick_size, self.instruments[b].tick_size)
                    if bid_a - ask_b < ARB_MIN_TICKS * tick - EPS:
                        continue
                    # Per leg-limiet; netto blijft gelijk dus groepslimiet speelt niet
                    pa, pb = positions.get(a, 0), positions.get(b, 0)
                    vol = min(bvol_a, avol_b, MAX_TAKE_VOLUME,
                              MAX_POS + pa, POSITION_LIMIT + pa,
                              MAX_POS - pb, POSITION_LIMIT - pb)
                    if vol <= 0:
                        continue
                    print(f"ARB: koop {vol} {b}@{ask_b}, verkoop {vol} {a}@{bid_a} "
                          f"(+{(bid_a - ask_b) * vol:.2f})")
                    if self.ioc(b, "bid", ask_b, vol):
                        self.ioc(a, "ask", bid_a, vol)
                    traded = True
        return traded

    def take_mispriced(self, tobs, fair, positions):
        traded = False
        for iid, inst in self.instruments.items():
            if iid not in tobs or iid not in fair:
                continue
            bid, bvol, ask, avol = tobs[iid]
            f, tick = fair[iid], inst.tick_size
            thresh = max(TAKE_MIN_TICKS * tick, TAKE_VOL_MULT * self.sigma.get(iid, 0.0))
            if ask <= f - thresh:
                vol = min(avol, MAX_TAKE_VOLUME, self.buy_room(iid, positions))
                if vol > 0:
                    print(f"TAKE: koop {vol} {iid}@{ask} (fair {f:.3f})")
                    traded |= self.ioc(iid, "bid", ask, vol)
            elif bid >= f + thresh:
                vol = min(bvol, MAX_TAKE_VOLUME, self.sell_room(iid, positions))
                if vol > 0:
                    print(f"TAKE: verkoop {vol} {iid}@{bid} (fair {f:.3f})")
                    traded |= self.ioc(iid, "ask", bid, vol)
        return traded

    # -- stap 3: quoten ------------------------------------------------------
    def target_quotes(self, iid, tob, f, positions):
        inst = self.instruments[iid]
        tick = inst.tick_size
        best_bid, _, best_ask, _ = tob
        pos = positions.get(iid, 0)
        net = self.group_net(iid, positions)

        edge = max(EDGE_MIN_TICKS * tick, EDGE_VOL_MULT * self.sigma.get(iid, 0.0))
        skew = -(SKEW_OWN_TICKS * pos / MAX_POS + SKEW_GROUP_TICKS * net / MAX_GROUP_NET) * tick

        bid = round_down(f - edge + skew, tick)
        ask = round_up(f + edge + skew, tick)
        # Passief blijven: nooit door het boek heen
        bid = min(bid, round(best_ask - tick, 6))
        ask = max(ask, round(best_bid + tick, 6))
        if bid >= ask:
            return None
        return (bid, min(QUOTE_VOLUME, self.buy_room(iid, positions)),
                ask, min(QUOTE_VOLUME, self.sell_room(iid, positions)))

    def sync_side(self, iid, side, price, volume, outstanding):
        tick = self.instruments[iid].tick_size
        tol = REQUOTE_TOL_TICKS * tick + EPS
        mine = [o for o in outstanding.values() if o.side == side]

        if volume > 0 and len(mine) == 1:
            o = mine[0]
            # Veilige kant: bestaande bid mag iets lager liggen, ask iets hoger
            safe = (price - tol <= o.price <= price + EPS) if side == "bid" \
                else (price - EPS <= o.price <= price + tol)
            if safe and volume / 2 <= o.volume <= volume:
                return

        for o in mine:
            self.ex.delete_order(iid, order_id=o.order_id)
        if volume > 0:
            resp = self.ex.insert_order(iid, price=price, volume=int(volume), side=side, order_type="limit")
            if not resp.success:
                self.log(f"lim-{iid}-{side}", f"[{iid}] {side} {volume}@{price} afgewezen: {resp.error_reason}")

    def quote(self, tobs, fair, positions):
        for iid in self.instruments:
            try:
                outstanding = self.ex.get_outstanding_orders(iid)
                q = None
                if iid in tobs and iid in fair:
                    q = self.target_quotes(iid, tobs[iid], fair[iid], positions)
                if q is None:
                    if outstanding:
                        self.ex.delete_orders(iid)
                    continue
                bid, bvol, ask, avol = q
                self.sync_side(iid, "bid", bid, bvol, outstanding)
                self.sync_side(iid, "ask", ask, avol, outstanding)
            except Exception as e:
                self.log(f"err-{iid}", f"[{iid}] fout bij quoten: {e}")

    # -- loop ----------------------------------------------------------------
    def pull_all(self):
        for iid in self.instruments:
            try:
                self.ex.delete_orders(iid)
            except Exception as e:
                print(f"[{iid}] delete_orders faalde: {e}")

    def step(self):
        tobs = {}
        for iid in self.instruments:
            tob = top_of_book(self.ex.get_last_price_book(iid))
            if tob:
                tobs[iid] = tob
        fair = self.compute_fairs(tobs)
        self.update_vol(fair)

        positions = self.ex.get_positions()
        if self.arbitrage(tobs, positions) | self.take_mispriced(tobs, fair, positions):
            positions = self.ex.get_positions()   # na IOC's opnieuw ophalen
        self.quote(tobs, fair, positions)
        self.analytics.on_step(self.ex, fair)
        return fair

    def report(self, fair):
        """Print analytics; geeft sessie-PnL terug (PnL sinds de bot startte)."""
        try:
            pnl = self.ex.get_pnl(valuations=fair) if fair else self.ex.get_pnl()
        except Exception:
            pnl = None
        nonzero = {k: v for k, v in self.ex.get_positions().items() if v}
        nets = {g: sum(nonzero.get(m, 0) for m in ms) for g, ms in self.groups.items()}
        nets = {g: n for g, n in nets.items() if n}
        return self.analytics.report(pnl, nonzero, nets)

    def run(self):
        print(f"Instrumenten: {list(self.instruments)}")
        print(f"Groepen: {self.groups}")
        last_report = 0.0
        try:
            while True:
                if not self.ex.is_connected():
                    print("Niet meer verbonden (teamgenoot ingelogd?). Stop.")
                    break
                try:
                    fair = self.step()
                except Exception as e:
                    self.log("step", f"fout in loop: {e}")
                    fair = {}

                if time.time() - last_report >= REPORT_EVERY:
                    last_report = time.time()
                    session = self.report(fair)
                    if session is not None and session < MAX_LOSS:
                        print(f"Sessie-PnL {session:.2f} < MAX_LOSS {MAX_LOSS}. Orders trekken en stoppen.")
                        break
                time.sleep(LOOP_SLEEP)
        except KeyboardInterrupt:
            print("Gestopt door gebruiker.")
        finally:
            if self.ex.is_connected():
                self.pull_all()


def main():
    exchange = Exchange()
    exchange.connect()
    MarketMaker(exchange).run()


if __name__ == "__main__":
    main()
