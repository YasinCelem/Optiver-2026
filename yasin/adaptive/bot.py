"""
Adaptive market maker.

FAIR VALUE
  - Stocks: the exchange "walls" (narrow stocks: 3000-lot ladder, wide: 1000-lot) sit around
    the true value; their midpoint is our fair value (top of book = other teams' small orders).
  - Lead-lag: NBIS/CRWV/SNDK copy AVGO/MU/NVDA ~1-3 s later; the expected catch-up is added.
  - Dual listings / futures / ETFs / options: no-arbitrage models on top of that.

QUOTING
  - Narrow stocks at the touch, wide stocks >= 2 ticks from the inventory-adjusted fair; one tick
    in front of the best competitor with real size (ignores 1-lot decoys); between the walls.
  - CLOCK: walls only move once a second at a fixed phase per stock -> learn it, and pull
    narrow-stock quotes from 120 ms before to 80 ms after (fills there lose ~1.4 ticks/lot).
  - Takes other teams' quotes 1 (narrow) / 2 (wide) ticks on the wrong side of fair.
  - Per-stock sizes (QUOTE_SIZES), skew on own position and on net exposure, where
    correlated pairs (PAIRS) and ETF-vs-constituents offset each other.

PROTECTION
  - Vol pause: fair moves >= JUMP_TICKS within JUMP_WINDOW_S -> stand aside on the stock and its
    pair partners; huge jumps (restart) must persist before we trust them. Refill block after fills.
  - Soft caps 40 (narrow) / 60 (wide), hard cut at 80 with a bounded IOC. Requests paced.
  - Request limiter, unwind / rebalance near limits, MAX_LOSS stop, news pause.

LEARNING (confidence weighting -- nothing switches on/off at a threshold)
  - Volatility: live estimate, with history (pretrain.py) as a small prior worth
    HIST_PRIOR_VOL_SAMPLES live samples that drops out when live data disagrees.
  - Adverse selection per side, counterparty scores: effect grows with live evidence.
  - Prediction model: effect = live-proven improvement over "no move" x live evidence.
  - Bandit on the edge multiplier after TIER3_FILLS live fills.
  - Tiers T0..T3 in the report are display only.

Everything learned is saved in data/brain.json. Raw logs go to data/raw/<session>/.
Every session appends a line to data/sessions.csv.

Run:  python yasin/adaptive/bot.py         Stop: Ctrl+C (orders are pulled).
Reset what it learned: delete yasin/adaptive/data/brain.json
"""
import cmath
import logging
import math
import os
import random
import sys
import time
from collections import defaultdict, deque
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C                                         # noqa: E402
from brain import Brain, DataLog                           # noqa: E402
from pricing import (bs_delta, bs_price, clip, fit_quadratic, implied_vol,  # noqa: E402
                     microprice, round_down, round_up, top_of_book, years_to_expiry)

logging.getLogger("client").setLevel("ERROR")

OPTION_TYPES = {"STOCK_OPTION", "INDEX_OPTION"}
FUTURE_TYPES = {"STOCK_FUTURE", "INDEX_FUTURE"}
KIND_PRIORITY = {"spot": 0, "etf": 1, "future": 2, "option": 3}


def type_name(inst):
    t = inst.instrument_type
    return "STOCK" if t is None else getattr(t, "name", str(t))


def kind_of(inst):
    t = type_name(inst)
    if t in OPTION_TYPES:
        return "option"
    if t in FUTURE_TYPES:
        return "future"
    if t == "INDEX_TRACKING_ETF":
        return "etf"
    return "spot"


def book_age_s(book):
    """Seconds since the book's timestamp, or None if it has none.

    A naive timestamp may be UTC or local time; take whichever reading is
    closer to now, so a timezone mismatch can't make every book look stale.
    """
    ts = getattr(book, "timestamp", None)
    if not isinstance(ts, datetime):
        return None
    if ts.tzinfo is not None:
        return (datetime.now(timezone.utc) - ts).total_seconds()
    ages = [(datetime.now() - ts).total_seconds(),
            (datetime.now(timezone.utc).replace(tzinfo=None) - ts).total_seconds()]
    return min(ages, key=abs)


def find_walls(book, tick):
    """The exchange's wall ladders. Narrow stocks: innermost level >= WALL_BIG lots
    (3000/2000/1000 ladder); wide stocks: >= WALL_SMALL (1000/750/500). Their midpoint is the
    exchange's own fair value; the top of book is usually small orders from other teams.
    Returns (wall_bid, wall_ask, narrow) or None."""
    def first(levels, vmin):
        return next((lv.price for lv in levels if lv.volume >= vmin), None)
    wb, wa, narrow = first(book.bids, C.WALL_BIG), first(book.asks, C.WALL_BIG), True
    if wb is None or wa is None:
        wb, wa, narrow = first(book.bids, C.WALL_SMALL), first(book.asks, C.WALL_SMALL), False
    if wb is None or wa is None:
        return None
    spread = (wa - wb) / tick
    if not 2 - 1e-6 <= spread <= C.MAX_WALL_SPREAD_TICKS:
        return None
    return wb, wa, narrow


def is_call(inst):
    k = inst.option_kind
    return getattr(k, "name", str(k)).upper().endswith("CALL")


def key(x):
    if isinstance(x, float) and x.is_integer():
        x = int(x)
    return str(x)


class Throttle:
    def __init__(self, every=5.0):
        self.every, self.last = every, {}

    def __call__(self, k, msg):
        now = time.time()
        if now - self.last.get(k, -1e18) >= self.every:
            self.last[k] = now
            print(msg)


class RateLimiter:
    """Sliding 1-second window on order requests (insert/delete). Exceeding the
    exchange's limit gets the whole team force-disconnected."""

    def __init__(self, per_s):
        self.per_s = per_s
        self.ts = deque()
        self.last = 0.0

    def available(self):
        now = time.time()
        while self.ts and now - self.ts[0] >= 1.0:
            self.ts.popleft()
        return self.per_s - len(self.ts)

    def take(self, n=1):
        if self.available() < n:
            return False
        self.ts.extend([time.time()] * n)
        return True

    def pace(self):
        """Keep at least ACTION_GAP_S between two requests (no bursts)."""
        wait = self.last + C.ACTION_GAP_S - time.time()
        if wait > 0:
            time.sleep(wait)
        self.last = time.time()


class AdaptiveBot:
    def __init__(self, exchange, data_dir=C.DATA_DIR):
        self.ex = exchange
        self.rl = RateLimiter(C.MAX_REQUESTS_PER_S)
        self.throttled = 0
        self.quote_offset = 0
        self.brain = Brain(data_dir)
        self.brain.s["sessions"] += 1
        self.log = DataLog(data_dir)
        self.rng = random.Random()
        self.warn = Throttle()

        tradable = exchange.get_tradable_instruments()
        self.instruments = self._select(tradable)
        self.kind = {i: kind_of(inst) for i, inst in self.instruments.items()}
        self.root = {i: self._root(i) for i in self.instruments}
        self.members = defaultdict(list)
        for i in sorted(self.instruments, key=lambda i: KIND_PRIORITY[self.kind[i]]):
            self.members[self.root[i]].append(i)
        self.index_def = self._index_defs()

        # live state (not persisted)
        self.tobs, self.mid, self.fair, self.fair_pre, self.theo = {}, {}, {}, {}, {}
        self.books, self.walls = {}, {}
        self.ref_hist = defaultdict(deque)
        self.guard_until, self.cooldown_until = {}, {}
        self.jumps = 0
        self.partners = self._pair_partners()
        self.narrow = {}                                  # iid -> True (3000-lot walls) / False (1000-lot)
        self.prev_wall_mid = {}
        self.jump_cand = {}
        self.chg_phase = defaultdict(lambda: deque(maxlen=30))   # local-clock fraction of a second at wall moves
        self.phase = {}                                   # learned clock phase per stock
        self.blackouts = 0
        self.debt = {}                                    # lead-lag: follower -> expected move still owed (ticks)
        self.ll_prev, self.debt_since, self.ll_t = {}, {}, None
        self.last_take = defaultdict(float)
        self.last_cut = {}
        self.order_t = {}                                 # (iid, side) -> when our current quote was placed
        self.affine, self.d1, self.U, self.eff = {}, {}, {}, {}
        self.vol_anchor, self.prev_t = {}, None
        self.last_unwind = defaultdict(float)
        self.basket = self._basket_units()
        self.flow, self.iflow = defaultdict(float), defaultdict(float)
        self.features = {}
        self.ioc_ids = set()
        self.check_age = C.MAX_BOOK_AGE_S is not None
        self.open_orders = {}
        self.pending_fills, self.pending_alpha, self.pending_names, self.pending_news = [], [], [], []
        self.last_alpha_sample = defaultdict(float)
        self.last_book_log = 0.0
        self.epoch, self.closed_epochs = {}, []
        self.pause_until = 0.0
        self.has_news = hasattr(exchange, "poll_new_social_media_feeds")
        self.surfaces = {}

        self.start_time = time.time()
        self.start_pnl = None
        self.session = None
        self.lots, self.notional = 0, 0.0
        self.mk_cash, self.mk_lots, self.mk_wins = 0.0, 0, 0

    # ================================================================== setup
    def _select(self, tradable):
        ids = C.INSTRUMENTS if C.INSTRUMENTS is not None else list(tradable)
        out = {}
        for i in ids:
            if i not in tradable or i in C.SKIP_IDS:
                continue
            if not C.TRADE_OPTIONS and kind_of(tradable[i]) == "option":
                continue
            out[i] = tradable[i]
        return out

    def _root(self, iid, depth=0):
        if iid in C.LINKED_ROOTS:
            return C.LINKED_ROOTS[iid]
        inst = self.instruments[iid]
        k = kind_of(inst)
        if k == "spot":
            return "S:" + key(getattr(inst, "instrument_group", None) or iid)
        base = inst.base_instrument_id
        if base and base in self.instruments and base != iid and depth < 5:
            return self._root(base, depth + 1)
        if k == "etf" and inst.index_id is not None:
            return "U:" + key(inst.index_id)
        if base is not None:
            return "U:" + key(base)
        if inst.index_id is not None:
            return "U:" + key(inst.index_id)
        return "U:" + iid

    def _index_defs(self):
        defs = {}
        for iid, inst in self.instruments.items():
            comp = getattr(inst, "index_constituents", None)
            if not comp:
                continue
            if all(c in self.instruments and self.kind[c] == "spot" for c in comp):
                defs[self.root[iid]] = (dict(comp), inst.index_divisor or 1.0)
            else:
                print(f"[{iid}] index constituents not all tradable; using books for its value")
        return defs

    def _pair_partners(self):
        """{iid: [(partner, corr), ...]} for configured correlated pairs that are both tradable."""
        out = defaultdict(list)
        for a, b, rho in C.PAIRS:
            if a in self.instruments and b in self.instruments:
                out[a].append((b, rho))
                out[b].append((a, rho))
        if out:
            print(f"Correlated pairs: { {k: v for k, v in out.items()} }")
        return dict(out)

    def ref_price(self, iid):
        """Wall mid when the exchange walls are visible, else microprice."""
        w = self.walls.get(iid)
        if w:
            return 0.5 * (w[0] + w[1])
        return microprice(self.tobs[iid])

    def _basket_units(self):
        """For each ETF/future on an index we know: the smallest whole-lot trade
        (n0 lots of it vs integer lots of each constituent) that is fully hedged."""
        out = {}
        for root, (comp, divisor) in self.index_def.items():
            for m in self.members[root]:
                if self.kind[m] == "option":
                    continue
                a, _ = self._affine(m)
                h = {c: a * w / divisor for c, w in comp.items()}
                for n0 in range(1, 51):
                    if all(abs(n0 * x - round(n0 * x)) <= 0.05 for x in h.values()):
                        q = {c: int(round(n0 * x)) for c, x in h.items() if round(n0 * x) > 0}
                        if q:
                            out[m] = (n0, q)
                            print(f"Basket for {m}: {n0} lots <-> {q}")
                        break
                else:
                    print(f"[{m}] no small whole-lot basket; basket arb off for it (hedging still works)")
        return out

    def _affine(self, iid, depth=0):
        """fair(iid) = a * U(root) + b for delta-one instruments."""
        inst, k = self.instruments[iid], self.kind[iid]
        if k == "spot":
            return 1.0, 0.0
        if k == "etf":
            return (inst.etf_multiplier or 1.0), (inst.etf_cash_comp or 0.0)
        if k == "future":
            carry = math.exp((inst.interest_rate or 0.0) * years_to_expiry(inst.expiry))
            base = inst.base_instrument_id
            if base in self.instruments and base != iid and self.kind[base] != "option" and depth < 5:
                a, b = self._affine(base, depth + 1)
                return carry * a, carry * b
            return carry, 0.0
        return None

    # ================================================================== market data
    def _check_jumps(self, now):
        """Fair moved >= JUMP_TICKS within JUMP_WINDOW_S: stand aside on the stock and its pair
        partners for JUMP_PAUSE_S. (Normal 1-second wall moves are handled by the clock blackout.)"""
        for iid, ref in self.mid.items():
            hist = self.ref_hist[iid]
            hist.append((now, ref))
            while len(hist) > 1 and now - hist[0][0] > C.JUMP_WINDOW_S:
                hist.popleft()
            tick = self.instruments[iid].tick_size
            lo = min(p for _, p in hist)
            hi = max(p for _, p in hist)
            if hi - lo >= C.JUMP_TICKS * tick - 1e-9:
                if now >= self.guard_until.get(iid, 0.0):
                    self.jumps += 1
                    print(f"JUMP {iid}: {(hi - lo) / tick:.1f} ticks in {C.JUMP_WINDOW_S:.0f}s -> standing aside "
                          f"{[iid] + [p for p, _ in self.partners.get(iid, [])]} for {C.JUMP_PAUSE_S:.0f}s")
                for j in [iid] + [p for p, _ in self.partners.get(iid, [])]:
                    self.guard_until[j] = max(self.guard_until.get(j, 0.0), now + C.JUMP_PAUSE_S)

    # ---------------------------------------------------------------- clock
    def _learn_phase(self, iid, now):
        """Walls move at a fixed fraction of each second; estimate it on our own clock
        (circular mean, trusted only when the observations cluster tightly)."""
        ch = self.chg_phase[iid]
        ch.append(now % 1.0)
        if len(ch) < C.CLOCK_MIN_SAMPLES:
            return
        z = sum(cmath.exp(2j * math.pi * f) for f in ch) / len(ch)
        if abs(z) >= C.CLOCK_MIN_R:
            # we notice a move up to one loop late: shift back by half a loop
            self.phase[iid] = (cmath.phase(z) / (2 * math.pi) - C.LOOP_SLEEP / 2) % 1.0
        else:
            self.phase.pop(iid, None)

    def in_blackout(self, iid, now):
        """True around the instant this stock's walls can move: nothing of ours may rest then."""
        if not C.CLOCK_AWARE or iid not in self.phase:
            return False
        if not self.narrow.get(iid, False) and not C.CLOCK_WIDE:
            return False
        d = (now - self.phase[iid]) % 1.0                 # seconds since the expected move
        pre = max(C.CLOCK_PRE_S, C.LOOP_SLEEP + 0.08)
        return d > 1.0 - pre or d < C.CLOCK_POST_S

    def cap(self, iid):
        """Soft position cap: per wall type for stocks, MAX_POS otherwise."""
        if iid in self.narrow:
            return C.SOFT_CAP_NARROW if self.narrow[iid] else C.SOFT_CAP_WIDE
        return C.MAX_POS

    def guarded(self, iid, now=None):
        return (now if now is not None else time.time()) < self.guard_until.get(iid, 0.0)

    def read_market(self, now, dt):
        self.tobs, self.mid, self.books, self.walls = {}, {}, {}, {}
        books = {iid: self.ex.get_last_price_book(iid) for iid in self.instruments}
        ages = {iid: book_age_s(b) for iid, b in books.items() if b is not None}
        ages = {iid: a for iid, a in ages.items() if a is not None}
        if self.check_age and ages and all(a > 60 for a in ages.values()):
            # Every book "minutes old" means our clock disagrees with the exchange's,
            # not that the whole market froze. Don't let that silence the bot.
            self.check_age = False
            print(f"WARNING: all books look >60s old (e.g. {max(ages.values()):.0f}s); "
                  f"clock mismatch? Stale-book check switched OFF.")
        for iid, book in books.items():
            age = ages.get(iid)
            if self.check_age and age is not None and age > C.MAX_BOOK_AGE_S:
                self.warn(f"stale-{iid}", f"[{iid}] book {age:.1f}s old -> not trading it")
                continue
            tob = top_of_book(book)
            if tob:
                self.tobs[iid] = tob
                self.books[iid] = book
                tick = self.instruments[iid].tick_size
                walls = find_walls(book, tick) if C.USE_WALLS and self.kind[iid] != "option" else None
                if walls:
                    wm = 0.5 * (walls[0] + walls[1])
                    prev = self.prev_wall_mid.get(iid)
                    if prev is not None and abs(wm - prev) > C.JUMP_CONFIRM_TICKS * tick:
                        # huge jump (e.g. exchange restart): believe it only once it persists
                        cand = self.jump_cand.get(iid)
                        if not cand or abs(cand[0] - wm) > 1e-9:
                            self.jump_cand[iid] = (wm, now)
                        if now - self.jump_cand[iid][1] < C.JUMP_CONFIRM_S:
                            self.guard_until[iid] = max(self.guard_until.get(iid, 0.0), now + C.JUMP_CONFIRM_S)
                            walls = None
                    if walls:
                        self.jump_cand.pop(iid, None)
                        if prev is not None and abs(wm - prev) > 1e-9 and abs(wm - prev) <= C.JUMP_CONFIRM_TICKS * tick:
                            self._learn_phase(iid, now)
                        self.prev_wall_mid[iid] = wm
                        self.narrow[iid] = walls[2]
                self.walls[iid] = walls
                # reference price: exchange wall mid (true value) when present, else plain mid
                self.mid[iid] = 0.5 * (walls[0] + walls[1]) if walls else 0.5 * (tob[0] + tob[2])
        if C.JUMP_GUARD:
            self._check_jumps(now)

        decay = 0.5 ** (dt / 5.0) if dt > 0 else 1.0
        for iid in self.instruments:
            self.flow[iid] *= decay
            self.iflow[iid] *= decay
            for t in self.ex.poll_new_trade_ticks(iid):
                side = getattr(t, "aggressor_side", None)
                sign = 1 if side == "bid" else -1 if side == "ask" else 0
                self.flow[iid] += sign * t.volume
                name = getattr(t, "buyer", None) if sign > 0 else getattr(t, "seller", None) if sign < 0 else None
                if name and iid in self.mid:
                    self.iflow[iid] += sign * t.volume * self.brain.name_score(name)
                    self.pending_names.append((now, iid, name, sign, self.mid[iid]))
                self.log.row("ticks", [now, iid, t.price, t.volume, side,
                                       getattr(t, "buyer", ""), getattr(t, "seller", "")])
        if len(self.pending_names) > 5000:
            self.pending_names = self.pending_names[-5000:]

        if self.has_news:
            try:
                posts = self.ex.poll_new_social_media_feeds()
            except Exception:
                self.has_news, posts = False, []
            for p in posts:
                self.log.row("news", [now, getattr(p, "post", ""), getattr(p, "meta_data", "")])
                self.pause_until = max(self.pause_until, now + self.brain.news_pause())
                self.pending_news.append((now, dict(self.mid)))
                print(f"NEWS: {str(getattr(p, 'post', ''))[:100]}")

    # ================================================================== fair values
    def compute_fairs(self):
        self.fair, self.fair_pre, self.theo, self.U, self.d1 = {}, {}, {}, {}, {}
        self.affine = {i: self._affine(i) for i in self.instruments if self.kind[i] != "option"}

        # roots priced from their own books first, then index roots built from constituents
        roots = sorted(self.members, key=lambda r: r in self.index_def)
        for root in roots:
            U = self._index_value(root) if root in self.index_def else None
            if U is None:
                U = self._book_value(root)
            if U is None:
                continue
            self.U[root] = U
            ref_a = self._ref_a(root)
            for iid in self.members[root]:
                if self.kind[iid] == "option":
                    continue
                a, b = self.affine[iid]
                self.d1[iid] = a / ref_a
                theo = a * U + b
                self.theo[iid] = theo
                if iid in self.tobs:
                    micro = self.ref_price(iid)
                    self.brain.update_basis(iid, micro - theo, abs(micro - theo) / max(abs(theo), 1e-9))
                if self.brain.model_trusted(iid):
                    self.fair_pre[iid] = theo + self.brain.basis(iid)
                elif iid in self.tobs:
                    self.fair_pre[iid] = self.ref_price(iid)
                    self.warn(f"distrust-{iid}", f"[{iid}] model off by "
                              f"{100 * self.brain.inst(iid)['rel_err']:.1f}% -> using its own book")

        self._build_surfaces()
        for iid in self.instruments:
            if self.kind[iid] == "option":
                self._price_option(iid)

        # short-term alpha on top of the model (only once it validates) + lead-lag debt
        self._update_debts(time.time())
        for iid, f in self.fair_pre.items():
            x = self._features(iid, f)
            self.features[iid] = x
            tick = self.instruments[iid].tick_size
            pred = self.brain.predict(type_name(self.instruments[iid]), x) if x else 0.0
            self.fair[iid] = f + C.ALPHA_WEIGHT * pred * tick + self.debt.get(iid, 0.0) * tick

    def _update_debts(self, now):
        """Lead-lag: a leader's move is 'owed' by its follower (beta x the move, in follower
        ticks); the follower moving the same way pays it back. Unpaid debt fades and times out.
        Follower moves on their own create no debt (fair value has no momentum by itself)."""
        dt = now - self.ll_t if self.ll_t is not None else 0.0
        self.ll_t = now
        for fol, (lead, beta) in C.LEADS.items():
            if fol not in self.fair_pre or lead not in self.fair_pre or not self.walls.get(fol) \
                    or not self.walls.get(lead):
                continue
            fa = self.fair_pre[lead] / self.instruments[lead].tick_size
            fb = self.fair_pre[fol] / self.instruments[fol].tick_size
            pa, pb = self.ll_prev.get(fol, (fa, fb))
            self.ll_prev[fol] = (fa, fb)
            d0 = self.debt.get(fol, 0.0) + beta * (fa - pa)
            df = fb - pb
            if d0 > 0:
                d = max(0.0, d0 - df) if df > 0 else d0
            elif d0 < 0:
                d = min(0.0, d0 - df) if df < 0 else d0
            else:
                d = 0.0
            d -= math.copysign(min(abs(d), C.LL_LEAK_TICKS_S * dt), d)
            d = clip(d, -C.LL_CAP_TICKS, C.LL_CAP_TICKS)
            if abs(d) >= C.LL_MIN_TICKS:
                if now - self.debt_since.setdefault(fol, now) > C.LL_TIMEOUT_S:
                    d = 0.0                                # the follower never moved: forget it
                    self.debt_since.pop(fol, None)
            else:
                self.debt_since.pop(fol, None)
            self.debt[fol] = d

    def _ref_a(self, root):
        for iid in self.members[root]:
            if iid in self.affine:
                return self.affine[iid][0] or 1.0
        return 1.0

    def _index_value(self, root):
        comp, divisor = self.index_def[root]
        if not all(c in self.fair_pre for c in comp):
            return None
        return sum(w * self.fair_pre[c] for c, w in comp.items()) / divisor

    def _book_value(self, root):
        """Liquidity-weighted underlying value implied by delta-one books."""
        def collect(only_trusted):
            num = den = 0.0
            for iid in self.members[root]:
                if self.kind[iid] == "option" or iid not in self.tobs:
                    continue
                if only_trusted and not self.brain.model_trusted(iid):
                    continue
                a, b = self.affine[iid]
                tob = self.tobs[iid]
                w = (tob[1] + tob[3]) * (0.5 if self.kind[iid] == "future" else 1.0)
                num += w * (self.ref_price(iid) - b) / a
                den += w
            return num / den if den > 0 else None
        v = collect(True)
        return v if v is not None else collect(False)

    def _underlying_of(self, iid):
        """(price, a_base) of an option's underlying."""
        base = self.instruments[iid].base_instrument_id
        if base in self.fair_pre:
            return self.fair_pre[base], self.affine.get(base, (1.0, 0.0))[0]
        root = self.root[iid]
        if root in self.U:
            return self.U[root], 1.0
        return None, None

    def _known_vol(self, iid):
        if C.OPTION_VOL_OVERRIDE:
            return C.OPTION_VOL_OVERRIDE
        inst = self.instruments[iid]
        if inst.index_volatility:
            return inst.index_volatility
        for m in self.members[self.root[iid]]:
            v = getattr(self.instruments[m], "index_volatility", None)
            if v:
                return v
        return None

    def _build_surfaces(self):
        """Per (root, expiry): flat vol from market IVs, or a quadratic smile once learned."""
        groups = defaultdict(list)
        for iid in self.instruments:
            if self.kind[iid] == "option":
                groups[(self.root[iid], key(self.instruments[iid].expiry))].append(iid)
        self.surfaces = {}
        for g, opts in groups.items():
            ivs = [(o, self.brain.inst(o)["iv"]) for o in opts
                   if self.brain.inst(o)["iv"] and self.brain.inst(o)["n_iv"] >= 5]
            if not ivs:
                continue
            vals = sorted(v for _, v in ivs)
            surf = {"flat": vals[len(vals) // 2], "lo": vals[0], "hi": vals[-1], "fit": None}
            mature = [(o, v) for o, v in ivs if self.brain.inst(o)["n_iv"] >= C.TIER1_BOOKS // 6]
            if len(mature) >= 3:
                S, _ = self._underlying_of(mature[0][0])
                if S:
                    xs = [math.log(self.instruments[o].strike / S) for o, _ in mature]
                    surf["fit"] = fit_quadratic(xs, [v for _, v in mature])
            self.surfaces[g] = surf

    def _price_option(self, iid):
        inst = self.instruments[iid]
        S, a_base = self._underlying_of(iid)
        if not S or not inst.strike:
            return
        T = years_to_expiry(inst.expiry)
        r = inst.interest_rate or 0.0
        call = is_call(inst)
        if iid in self.tobs:
            bid, _, ask, _ = self.tobs[iid]
            m = 0.5 * (bid + ask)
            if m > 0 and (ask - bid) / m < 0.5:
                iv = implied_vol(call, m, S, inst.strike, T, r)
                if iv:
                    self.brain.update_iv(iid, iv)
        sigma = self._known_vol(iid)
        if sigma is None:
            surf = self.surfaces.get((self.root[iid], key(inst.expiry)))
            if surf is None:
                return
            sigma = surf["flat"]
            if surf["fit"]:
                c = surf["fit"]
                k = math.log(inst.strike / S)
                sigma = clip(c[0] + c[1] * k + c[2] * k * k, 0.8 * surf["lo"], 1.2 * surf["hi"])
        theo = bs_price(call, S, inst.strike, T, r, sigma)
        if theo is None:
            return
        self.theo[iid] = theo
        self.fair_pre[iid] = max(theo, inst.tick_size)
        self.d1[iid] = bs_delta(call, S, inst.strike, T, r, sigma) * a_base / self._ref_a(self.root[iid])

    def _features(self, iid, f):
        if iid not in self.tobs:
            return None
        bid, bv, ask, av = self.tobs[iid]
        tick = self.instruments[iid].tick_size
        imb = (bv - av) / (bv + av) if bv + av else 0.0
        return [imb, clip(self.flow[iid] / 20.0, -3, 3),
                clip((microprice(self.tobs[iid]) - f) / tick, -5, 5),
                clip(self.iflow[iid] / 20.0, -3, 3)]

    # ================================================================== learning
    def learn(self, now, dt):
        # vol from moves over VOL_SAMPLE_S, not per loop: a microprice that flickers
        # between ticks adds noise^2/dt, which is 20x smaller at 5s than at 0.25s.
        for iid, f in self.fair_pre.items():
            self.brain.count_book(iid)
            anchor = self.vol_anchor.get(iid)
            if anchor is None:
                self.vol_anchor[iid] = (now, f)
            elif now - anchor[0] >= C.VOL_SAMPLE_S:
                self.brain.update_vol(iid, f - anchor[1], now - anchor[0])
                self.vol_anchor[iid] = (now, f)

        # sample features now, label them ALPHA_HORIZON_S later
        for iid, x in self.features.items():
            if x and iid in self.mid and now - self.last_alpha_sample[iid] >= C.ALPHA_SAMPLE_EVERY_S:
                self.last_alpha_sample[iid] = now
                self.pending_alpha.append((now, iid, x, self.mid[iid]))

        keep = []
        for p in self.pending_fills:
            t, iid, sign, price, vol, aggressive, ep = p
            ref = self.fair.get(iid, self.mid.get(iid))
            if now - t < C.MARKOUT_S or ref is None:
                if now - t < 120:
                    keep.append(p)
                continue
            cash = (ref - price) * sign * vol
            if not aggressive:
                self.brain.record_markout(iid, "bid" if sign > 0 else "ask",
                                          (ref - price) * sign / self.instruments[iid].tick_size)
            if ep is not None:
                ep["cash"] += cash
            self.mk_cash += cash
            self.mk_lots += vol
            self.mk_wins += vol if cash > 0 else 0
        self.pending_fills = keep

        self.pending_alpha = self._resolve(self.pending_alpha, now, C.ALPHA_HORIZON_S, lambda p: (
            self.brain.add_sample(type_name(self.instruments[p[1]]), p[2],
                                  clip((self.mid[p[1]] - p[3]) / self.instruments[p[1]].tick_size, -10, 10))))
        self.pending_names = self._resolve(self.pending_names, now, C.MARKOUT_S, lambda p: (
            self.brain.record_name(p[2], (self.mid[p[1]] - p[4]) * p[3] / self.instruments[p[1]].tick_size)))

        keep = []
        for t, mids in self.pending_news:
            if now - t < C.MARKOUT_S:
                keep.append((t, mids))
                continue
            ratios = [abs(self.mid[i] - m0) / (self.brain.sigma_1s(i) * math.sqrt(C.MARKOUT_S)
                                                + 0.5 * self.instruments[i].tick_size)
                      for i, m0 in mids.items() if i in self.mid]
            if ratios:
                self.brain.record_news_impact(sum(ratios) / len(ratios))
        self.pending_news = keep

        self._bandit_epochs(now)

    def _resolve(self, pending, now, horizon, fn):
        keep = []
        for p in pending:
            if now - p[0] < horizon:
                keep.append(p)
            elif p[1] in self.mid:
                fn(p)
            elif now - p[0] < 120:
                keep.append(p)
        return keep

    def _bandit_epochs(self, now):
        for iid in self.instruments:
            ep = self.epoch.get(iid)
            if ep is None or now - ep["start"] >= C.BANDIT_EPOCH_S:
                if ep is not None:
                    ep["end"] = now
                    self.closed_epochs.append((iid, ep))
                self.epoch[iid] = {"arm": self.brain.bandit_choose(iid, self.rng),
                                   "start": now, "end": None, "cash": 0.0}
        keep = []
        for iid, ep in self.closed_epochs:
            if now >= ep["end"] + C.MARKOUT_S + 1:
                minutes = max(1e-6, (ep["end"] - ep["start"]) / 60.0)
                self.brain.bandit_record(iid, ep["arm"], ep["cash"] / minutes)
            else:
                keep.append((iid, ep))
        self.closed_epochs = keep

    # ================================================================== fills
    def drain_fills(self, iid):
        now = time.time()
        out = []
        for t in self.ex.poll_new_trades(iid):
            sign = 1 if t.side == "bid" else -1
            aggressive = t.order_id in self.ioc_ids
            f = self.fair.get(iid)
            tick = self.instruments[iid].tick_size
            edge = (f - t.price) * sign / tick if f is not None else None
            st = self.brain.inst(iid)
            st["fills"] += 1
            if not aggressive:
                st["passive_fills"] += 1
                self.cooldown_until[(iid, t.side)] = now + C.FILL_COOLDOWN_S
                self.order_t.pop((iid, t.side), None)
            self.lots += t.volume
            self.notional += abs(t.price * t.volume)
            ep = None if aggressive else self.epoch.get(iid)
            self.pending_fills.append((now, iid, sign, t.price, t.volume, aggressive, ep))
            self.log.row("fills", [now, iid, t.side, t.price, t.volume, f, edge, aggressive,
                                   ep["arm"] if ep else ""])
            out.append(t)
        return out

    # ================================================================== risk helpers
    def root_deltas(self, positions):
        d = defaultdict(float)
        for iid, dl in self.d1.items():
            d[self.root[iid]] += positions.get(iid, 0) * dl
        return d

    def refresh_risk(self, positions):
        """Raw delta per root (used for hard limits) and effective exposure
        (used for skew / unwinding / hedging), where an ETF position and its
        constituent stocks offset each other."""
        raw = self.root_deltas(positions)
        eff = defaultdict(float, raw)
        for R, (comp, divisor) in self.index_def.items():
            ref_a = self._ref_a(R)
            pos_u = raw.get(R, 0.0) * ref_a                  # exposure in index units
            num = den = 0.0
            for c, w in comp.items():
                h = w / divisor
                eff[self.root[c]] += pos_u * h               # stock lots the index position represents
                num += raw.get(self.root[c], 0.0) * h
                den += h * h
            if den > 0:
                eff[R] += (num / den) / ref_a                # index units the stocks represent
        # correlated pairs: a partner's position counts PAIR_NET_FACTOR x corr x vol ratio
        for iid, plist in self.partners.items():
            ra = self.root[iid]
            for p, rho in plist:
                rb = self.root[p]
                if rb != ra:
                    eff[ra] += self.pair_weight(iid, p, rho) * raw.get(rb, 0.0)
        self.eff = eff
        return raw

    def pair_weight(self, iid, partner, rho):
        """Lots of `iid` that one lot of `partner` behaves like (volatility-matched)."""
        sa, sb = self.brain.sigma_1s(iid), self.brain.sigma_1s(partner)
        if sa > 0 and sb > 0:
            ratio = sb / sa
        elif iid in self.mid and partner in self.mid and self.mid[iid] > 0:
            ratio = self.mid[partner] / self.mid[iid]
        else:
            ratio = 1.0
        return C.PAIR_NET_FACTOR * rho * ratio

    @staticmethod
    def _curve(u):
        u = clip(u, -1.5, 1.5)
        return u + C.SKEW_CONVEXITY * u ** 3

    def open_vol(self, iid, side):
        """Volume of our resting orders on this side (they can still fill)."""
        return sum(int(o.volume) for o in self.open_orders.get(iid, {}).values() if o.side == side)

    def room(self, iid, side, positions, deltas, use_delta=True, include_open=True):
        """Lots we may add on `side`. include_open=False when the resting order on
        that side is about to be replaced (quoting), so it shouldn't count twice."""
        r = self._room(iid, side, positions, deltas, use_delta)
        if include_open:
            r -= self.open_vol(iid, side)
        return max(0, r)

    def _room(self, iid, side, positions, deltas, use_delta):
        pos = positions.get(iid, 0)
        cap = self.cap(iid)
        if side == "bid":
            r, dchg = min(cap - pos, C.POSITION_LIMIT - pos), self.d1.get(iid, 0.0)
        else:
            r, dchg = min(cap + pos, C.POSITION_LIMIT + pos), -self.d1.get(iid, 0.0)
        if use_delta and abs(dchg) > 1e-9:
            rd = deltas.get(self.root[iid], 0.0)
            root_cap = min(C.MAX_ROOT_DELTA, cap) if iid in self.narrow else C.MAX_ROOT_DELTA
            lim = (root_cap - rd) / dchg if dchg > 0 else (root_cap + rd) / -dchg
            r = min(r, math.floor(lim + 1e-9))
        return max(0, int(r))

    def ioc(self, iid, side, price, volume, wait=False):
        volume = int(volume)
        if volume <= 0:
            return 0
        if not self.rl.take():
            self.throttled += 1
            return 0
        self.rl.pace()
        resp = self.ex.insert_order(iid, price=price, volume=volume, side=side, order_type="ioc")
        if not resp.success:
            self.warn(f"ioc-{iid}", f"[{iid}] IOC {side} {volume}@{price} rejected: {resp.error_reason}")
            return 0
        self.ioc_ids.add(resp.order_id)
        if not wait:
            return volume
        filled = 0
        for attempt in range(2):
            filled += sum(t.volume for t in self.drain_fills(iid) if t.order_id == resp.order_id)
            if filled:
                break
            time.sleep(0.05)
        return filled

    # ================================================================== trading
    def arbitrage(self, positions):
        traded = False
        for root, mem in self.members.items():
            legs = [m for m in mem if self.kind[m] != "option" and m in self.tobs
                    and self.brain.model_trusted(m)]
            done = False
            for a in legs:
                for b in legs:
                    if a == b or done:
                        continue
                    aa, ba = self.affine[a]
                    ab, bb = self.affine[b]
                    ratio = aa / ab                     # lots of b per lot of a
                    if not 0.5 <= ratio <= 2.0:
                        continue
                    bid_a, bv_a, _, _ = self.tobs[a]
                    _, _, ask_b, av_b = self.tobs[b]
                    profit = ((bid_a - ba) / aa - (ask_b - bb) / ab) * aa   # per lot of a
                    tick = max(self.instruments[a].tick_size, self.instruments[b].tick_size)
                    pa, pb = positions.get(a, 0), positions.get(b, 0)
                    # Offsetting legs near their limits (long a, short b) block quoting on
                    # both; flattening them is worth paying a little for.
                    rebalance = (pa > 0 > pb and max(pa, -pb) > C.UNWIND_TRIGGER_FRAC * C.MAX_POS)
                    min_profit = -C.REBALANCE_COST_TICKS * tick if rebalance else C.ARB_MIN_TICKS * tick
                    if profit < min_profit - 1e-9:
                        continue
                    sell_a = min(C.MAX_POS + pa, C.POSITION_LIMIT + pa) - self.open_vol(a, "ask")
                    buy_b = min(C.MAX_POS - pb, C.POSITION_LIMIT - pb) - self.open_vol(b, "bid")
                    vol_a = min(bv_a, int(av_b / ratio), C.MAX_TAKE_VOLUME, sell_a, int(buy_b / ratio))
                    if profit < C.ARB_MIN_TICKS * tick - 1e-9:          # pure rebalance: don't overshoot
                        vol_a = min(vol_a, C.UNWIND_CHUNK, pa, int(-pb / ratio))
                    vol_b = int(round(vol_a * ratio))
                    if vol_a < 1 or vol_b < 1:
                        continue
                    tag = "ARB" if profit >= C.ARB_MIN_TICKS * tick - 1e-9 else "REBALANCE"
                    print(f"{tag} {root}: buy {vol_b} {b}@{ask_b}, sell {vol_a} {a}@{bid_a} ({profit * vol_a:+.2f})")
                    filled_b = self.ioc(b, "bid", ask_b, vol_b, wait=True)
                    if filled_b > 0:
                        self.ioc(a, "ask", bid_a, min(vol_a, max(1, int(round(filled_b / ratio)))))
                    traded = done = True
        return traded

    def basket_arb(self, positions):
        """ETF (or index future) vs its constituent stocks, top of book only."""
        if not C.BASKET_ARB:
            return False
        traded = False
        for m, (n0, q) in self.basket.items():
            if m not in self.tobs or any(c not in self.tobs for c in q) or not self.brain.model_trusted(m):
                continue
            _, b = self.affine[m]
            tick = self.instruments[m].tick_size
            bid_m, bv_m, ask_m, av_m = self.tobs[m]
            basket_ask = sum(n * self.tobs[c][2] for c, n in q.items())
            basket_bid = sum(n * self.tobs[c][0] for c, n in q.items())
            for side_m, price_m, avail_m, profit in (
                    ("ask", bid_m, bv_m, n0 * (bid_m - b) - basket_ask),     # sell ETF, buy stocks
                    ("bid", ask_m, av_m, basket_bid - n0 * (ask_m - b))):    # buy ETF, sell stocks
                if profit < C.ARB_MIN_TICKS * tick * n0 - 1e-9:
                    continue
                side_c = "bid" if side_m == "ask" else "ask"
                k = min(avail_m // n0, max(1, C.MAX_TAKE_VOLUME // n0),
                        self.room(m, side_m, positions, {}, use_delta=False) // n0)
                for c, n in q.items():
                    lvl_vol = self.tobs[c][3] if side_c == "bid" else self.tobs[c][1]
                    k = min(k, lvl_vol // n, self.room(c, side_c, positions, {}, use_delta=False) // n)
                if k < 1:
                    continue
                print(f"BASKET: {'sell' if side_m == 'ask' else 'buy'} {k * n0} {m}@{price_m} vs "
                      f"{ {c: k * n for c, n in q.items()} } (+{profit * k:.2f})")
                filled = self.ioc(m, side_m, price_m, k * n0, wait=True)
                if filled > 0:
                    for c, n in q.items():
                        qty = int(round(filled * n / n0))
                        price_c = self.tobs[c][2] if side_c == "bid" else self.tobs[c][0]
                        self.ioc(c, side_c, price_c, qty)
                traded = True
                break
        return traded

    def unwind(self, positions, deltas, now):
        """Exposure near the limit: cross the spread in the cheapest linked instrument,
        but only when that costs little. Roots with options use hedge() instead."""
        traded = False
        trigger = C.UNWIND_TRIGGER_FRAC * C.MAX_ROOT_DELTA
        for root, mem in self.members.items():
            if any(self.kind[m] == "option" for m in mem):
                continue
            x = self.eff.get(root, 0.0)
            if abs(x) <= trigger or now - self.last_unwind[root] < C.UNWIND_COOLDOWN_S:
                continue
            side = "ask" if x > 0 else "bid"
            best = None
            cands = [(m, self.d1.get(m, 0.0)) for m in mem]
            for m in mem:      # pair partners can hedge too, unless that pushes their own exposure the wrong way
                for p, rho in self.partners.get(m, []):
                    xp = self.eff.get(self.root[p], 0.0)
                    if xp * x >= 0 or abs(xp) < 0.5 * trigger:
                        cands.append((p, self.pair_weight(m, p, rho) * self.d1.get(p, 0.0)))
            for m, dl in cands:
                if self.kind[m] == "option" or m not in self.tobs or m not in self.fair or dl <= 1e-9:
                    continue
                bid, bv, ask, av = self.tobs[m]
                tick = self.instruments[m].tick_size
                price, avail = (bid, bv) if side == "ask" else (ask, av)
                cost = (self.fair[m] - price) / tick if side == "ask" else (price - self.fair[m]) / tick
                if cost > C.UNWIND_MAX_COST_TICKS:
                    continue
                vol = min(C.UNWIND_CHUNK, avail, self.room(m, side, positions, deltas, use_delta=False),
                          math.ceil((abs(x) - trigger) / dl))
                if vol > 0 and (best is None or cost / dl < best[0]):
                    best = (cost / dl, m, price, vol, cost)
            if best:
                _, m, price, vol, cost = best
                print(f"UNWIND {root}: exposure {x:+.1f} -> {side} {vol} {m}@{price} (cost {cost:.1f} ticks/lot)")
                self.last_unwind[root] = now
                traded |= self.ioc(m, side, price, vol) > 0
        return traded

    def take_mispriced(self, positions, deltas):
        traded = False
        now = time.time()
        for iid, inst in self.instruments.items():
            if (iid not in self.tobs or iid not in self.fair or not self.brain.model_trusted(iid)
                    or self.guarded(iid, now)):
                continue
            f, tick = self.fair[iid], inst.tick_size
            walls = self.walls.get(iid)
            if walls:
                # hit other teams' quotes that sit on the wrong side of our inventory-adjusted fair
                # (incl. the follower's wall when the lead-lag says it's about to move)
                if now - self.last_take[iid] < C.TAKE_COOLDOWN_S:
                    continue
                r = f - self.skew_ticks(iid, positions) * tick
                need = (C.TAKE_EDGE_NARROW_TICKS if walls[2] else C.TAKE_EDGE_WIDE_TICKS) * tick
                big = C.TAKE_SIZE_BIG if abs(self.debt.get(iid, 0.0)) >= C.LL_BIG_TICKS else C.MAX_TAKE_VOLUME
                tb, ta = self.competitor_top(iid, "bid"), self.competitor_top(iid, "ask")
                if tb and tb[0] - r >= need - 1e-9:
                    vol = min(tb[1], big, self.room(iid, "ask", positions, deltas))
                    if vol > 0:
                        print(f"TAKE: sell {vol} {iid}@{tb[0]} (fair {f:.3f}, adj {r:.3f})")
                        self.last_take[iid] = now
                        traded |= self.ioc(iid, "ask", tb[0], vol) > 0
                elif ta and r - ta[0] >= need - 1e-9:
                    vol = min(ta[1], big, self.room(iid, "bid", positions, deltas))
                    if vol > 0:
                        print(f"TAKE: buy {vol} {iid}@{ta[0]} (fair {f:.3f}, adj {r:.3f})")
                        self.last_take[iid] = now
                        traded |= self.ioc(iid, "bid", ta[0], vol) > 0
                continue
            bid, bv, ask, av = self.tobs[iid]
            thresh = max(C.TAKE_MIN_TICKS * tick, C.TAKE_VOL_MULT * self.brain.sigma_1s(iid))
            # little vol evidence yet: only obvious mispricings (fades with confidence)
            thresh = max(thresh, (1 - self.brain.vol_conf(iid)) * (ask - bid))
            if ask <= f - thresh:
                vol = min(av, C.MAX_TAKE_VOLUME, self.room(iid, "bid", positions, deltas))
                if vol > 0:
                    print(f"TAKE: buy {vol} {iid}@{ask} (fair {f:.3f})")
                    traded |= self.ioc(iid, "bid", ask, vol) > 0
            elif bid >= f + thresh:
                vol = min(bv, C.MAX_TAKE_VOLUME, self.room(iid, "ask", positions, deltas))
                if vol > 0:
                    print(f"TAKE: sell {vol} {iid}@{bid} (fair {f:.3f})")
                    traded |= self.ioc(iid, "ask", bid, vol) > 0
        return traded

    def hard_cap_cut(self, positions, now):
        """|position| >= HARD_CAP: bounded IOC down to half the soft cap (at most CUT_SLIP_TICKS
        through the wall). Waits 1 s between cuts so the position can update."""
        traded = False
        for iid in self.instruments:
            pos = positions.get(iid, 0)
            walls = self.walls.get(iid)
            if abs(pos) < C.HARD_CAP or not walls:
                continue
            t0, p0 = self.last_cut.get(iid, (0.0, None))
            if p0 == pos and now - t0 < 1.0:
                continue
            tick = self.instruments[iid].tick_size
            vol = abs(pos) - self.cap(iid) // 2
            side = "ask" if pos > 0 else "bid"
            price = (round_down(walls[0] - C.CUT_SLIP_TICKS * tick, tick) if pos > 0
                     else round_up(walls[1] + C.CUT_SLIP_TICKS * tick, tick))
            print(f"HARD CAP {iid}: position {pos} -> {side} {vol} up to {price}")
            self.last_cut[iid] = (now, pos)
            traded |= self.ioc(iid, side, price, vol) > 0
        return traded

    def hedge(self, positions, deltas):
        traded = False
        for root, mem in self.members.items():
            if not any(self.kind[m] == "option" for m in mem):
                continue
            delta = self.eff.get(root, 0.0)
            if abs(delta) <= C.HEDGE_TRIGGER:
                continue
            cands = [m for m in mem if self.kind[m] != "option" and m in self.tobs and abs(self.d1.get(m, 0)) > 1e-9]
            if not cands:
                continue
            h = min(cands, key=lambda m: (self.tobs[m][2] - self.tobs[m][0]) / self.mid[m])
            side = "ask" if delta > 0 else "bid"
            bid, bv, ask, av = self.tobs[h]
            price, avail = (bid, bv) if side == "ask" else (ask, av)
            vol = min(int((abs(delta) - C.HEDGE_TARGET) / abs(self.d1[h])), avail, C.MAX_TAKE_VOLUME,
                      self.room(h, side, positions, deltas, use_delta=False))
            if vol > 0:
                print(f"HEDGE {root}: delta {delta:+.1f} -> {side} {vol} {h}@{price}")
                traded |= self.ioc(h, side, price, vol) > 0
        return traded

    def _others(self, iid, side):
        """[(price, volume)] on `side` with our own resting volume removed, best first."""
        book = self.books.get(iid)
        if not book:
            return []
        mine = defaultdict(int)
        for o in self.open_orders.get(iid, {}).values():
            if o.side == side:
                mine[round(o.price, 6)] += int(o.volume)
        out = []
        for lv in (book.bids if side == "bid" else book.asks):
            v = lv.volume - mine.get(round(lv.price, 6), 0)
            if v > 0:
                out.append((lv.price, v))
        return out

    def competitor_best(self, iid, side):
        """Best competing price with real size: a level only counts once >= COMP_VOL lots rest at
        or better than it, so a 1-lot decoy can't bait us into giving up edge."""
        cum = 0
        for price, v in self._others(iid, side):
            cum += v
            if cum >= C.COMP_VOL:
                return price
        return None

    def competitor_top(self, iid, side):
        """(price, volume) of the best level that isn't ours, or None."""
        lv = self._others(iid, side)
        return lv[0] if lv else None

    def skew_ticks(self, iid, positions):
        """Inventory lean in ticks (positive = we're long -> lower our prices): own position plus
        net exposure of its root (pairs / ETF netting), each on a linear + cubic curve."""
        cap = self.cap(iid)
        pos = positions.get(iid, 0)
        exposure = self.eff.get(self.root[iid], 0.0)
        dl = clip(self.d1.get(iid, 1.0), -1.0, 1.0)
        root_cap = cap if iid in self.narrow else C.MAX_ROOT_DELTA
        return (C.SKEW_OWN * self._curve(pos / cap)
                + C.SKEW_ROOT * self._curve(exposure / root_cap) * dl)

    def target_quotes(self, iid, positions, deltas, now=None):
        now = time.time() if now is None else now
        inst = self.instruments[iid]
        tick = inst.tick_size
        best_bid, _, best_ask, _ = self.tobs[iid]
        f = self.fair[iid]
        size = (C.OPTION_QUOTE_VOLUME if self.kind[iid] == "option"
                else C.QUOTE_SIZES.get(iid, C.QUOTE_VOLUME))
        if size <= 0:
            return None
        walls = self.walls.get(iid)

        if walls:
            # ---- stocks with walls: teammate-style. Edge measured from the inventory-adjusted fair
            # (reservation); penny the best competitor with real size; stay between the walls.
            wb, wa, narrow = walls
            edge = (C.MIN_EDGE_NARROW_TICKS if narrow else C.MIN_EDGE_WIDE_TICKS) * tick
            edge *= self.epoch[iid]["arm"] if iid in self.epoch else 1.0
            e_bid = max(0.0, edge + self.brain.adverse_extra_ticks(iid, "bid") * tick)
            e_ask = max(0.0, edge + self.brain.adverse_extra_ticks(iid, "ask") * tick)
            r = f - self.skew_ticks(iid, positions) * tick
            bid = round_down(r - e_bid, tick)
            ask = round_up(r + e_ask, tick)
            no_bid, no_ask = bid < wb - 1e-9, ask > wa + 1e-9      # would be behind the wall: don't quote
            if C.QUOTE_INSIDE:
                cb, ca = self.competitor_best(iid, "bid"), self.competitor_best(iid, "ask")
                bid = min(bid, round_down((cb if cb is not None else wb) + tick, tick))
                ask = max(ask, round_up((ca if ca is not None else wa) - tick, tick))
            ask = max(min(ask, wa), round_up(wb + tick, tick))
            bid = min(max(bid, wb), round_down(wa - tick, tick))
            if bid >= ask:
                bid = round_down(ask - tick, tick)
        else:
            # ---- no walls (other instrument types): volatility-based edge around fair
            no_bid = no_ask = False
            edge = max(C.EDGE_MIN_TICKS * tick, C.EDGE_VOL_MULT * self.brain.sigma_1s(iid))
            edge = max(edge, (1 - self.brain.vol_conf(iid)) * C.COLD_SPREAD_FRAC * (best_ask - best_bid) / 2)
            edge *= self.epoch[iid]["arm"] if iid in self.epoch else 1.0
            e_bid = max(C.EDGE_MIN_TICKS * tick, edge + self.brain.adverse_extra_ticks(iid, "bid") * tick)
            e_ask = max(C.EDGE_MIN_TICKS * tick, edge + self.brain.adverse_extra_ticks(iid, "ask") * tick)
            skew = -max(tick, edge) * self.skew_ticks(iid, positions)
            bid = round_down(f - e_bid + skew, tick)
            ask = round_up(f + e_ask + skew, tick)
            bid = min(bid, round_down(best_ask - tick, tick))
            ask = max(ask, round_up(best_bid + tick, tick))
            if bid >= ask:
                return None

        vb = min(size, self.room(iid, "bid", positions, deltas, include_open=False))
        va = min(size, self.room(iid, "ask", positions, deltas, include_open=False))
        if no_bid or now < self.cooldown_until.get((iid, "bid"), 0.0):   # just filled: don't refill yet
            vb = 0
        if no_ask or now < self.cooldown_until.get((iid, "ask"), 0.0):
            va = 0
        if vb < C.MIN_QUOTE_SIZE:
            vb = 0
        if va < C.MIN_QUOTE_SIZE:
            va = 0
        pl = getattr(inst, "price_limit", None)
        if bid < tick or (pl and bid < pl.min):
            vb = 0
        if pl and ask > pl.max:
            va = 0
        return bid, vb, ask, va

    def sync_side(self, iid, side, price, volume, outstanding):
        tick = self.instruments[iid].tick_size
        tol = C.REQUOTE_TOL_TICKS * tick + 1e-9
        mine = [o for o in outstanding.values() if o.side == side]
        if volume > 0 and len(mine) == 1:
            o = mine[0]
            safe = (price - tol <= o.price <= price + 1e-9) if side == "bid" else (price - 1e-9 <= o.price <= price + tol)
            if safe and volume / 2 <= o.volume <= volume:
                return
            # less aggressive than wanted (never more): keep it until MIN_REQUOTE_S old
            calmer = o.price <= price + 1e-9 if side == "bid" else o.price >= price - 1e-9
            age = time.time() - self.order_t.get((iid, side), 0.0)
            if calmer and o.volume <= volume and age < C.MIN_REQUOTE_S:
                return
        # reserve the whole replace up front: never delete without being able to re-insert
        if not self.rl.take(len(mine) + (1 if volume > 0 else 0)):
            self.throttled += 1
            return
        for o in mine:
            self.rl.pace()
            resp = self.ex.delete_order(iid, order_id=o.order_id)
            if resp is not None and not getattr(resp, "success", True):
                self.warn(f"del-{iid}-{side}", f"[{iid}] delete {side} {o.order_id} failed: "
                          f"{getattr(resp, 'error_reason', '?')} -> not re-quoting this loop")
                return
        if mine and any(o.side == side for o in self.ex.get_outstanding_orders(iid).values()):
            # Old order still live (failed or racing cancel): never stack a second one.
            return
        self.order_t.pop((iid, side), None)
        if volume > 0:
            self.rl.pace()
            resp = self.ex.insert_order(iid, price=price, volume=int(volume), side=side, order_type="limit")
            if not resp.success:
                self.warn(f"lim-{iid}-{side}", f"[{iid}] {side} {volume}@{price} rejected: {resp.error_reason}")
            else:
                self.order_t[(iid, side)] = time.time()

    def quote(self, positions, deltas, paused):
        # rotate the starting instrument so a tight request budget is shared fairly
        ids = list(self.instruments)
        self.quote_offset = (self.quote_offset + 1) % max(1, len(ids))
        for iid in ids[self.quote_offset:] + ids[:self.quote_offset]:
            try:
                outstanding = self.open_orders.get(iid)
                if outstanding is None:
                    outstanding = self.ex.get_outstanding_orders(iid)
                q = None
                now = time.time()
                blackout = self.in_blackout(iid, now)
                if not paused and not blackout and not self.guarded(iid, now) and iid in self.tobs and iid in self.fair:
                    q = self.target_quotes(iid, positions, deltas, now)
                if q is None or (q[1] == 0 and q[3] == 0):
                    if outstanding and self.rl.take():
                        if blackout:
                            self.blackouts += 1
                        self.rl.pace()
                        self.ex.delete_orders(iid)
                        self.order_t.pop((iid, "bid"), None)
                        self.order_t.pop((iid, "ask"), None)
                    continue
                bid, vb, ask, va = q
                self.sync_side(iid, "bid", bid, vb, outstanding)
                self.sync_side(iid, "ask", ask, va, outstanding)
            except Exception as e:
                self.warn(f"q-{iid}", f"[{iid}] quoting error: {e}")

    # ================================================================== loop
    def step(self):
        now = time.time()
        dt = now - self.prev_t if self.prev_t else 0.0
        self.prev_t = now

        self.read_market(now, dt)
        self.compute_fairs()
        self.learn(now, dt)

        paused = now < self.pause_until
        self.open_orders = {iid: self.ex.get_outstanding_orders(iid) for iid in self.instruments}
        positions = self.ex.get_positions()
        deltas = self.refresh_risk(positions)
        actions = [lambda: self.arbitrage(positions), lambda: self.basket_arb(positions)]
        if not paused:
            actions.append(lambda: self.take_mispriced(positions, deltas))
        actions += [lambda: self.hedge(positions, deltas), lambda: self.unwind(positions, deltas, now),
                    lambda: self.hard_cap_cut(positions, now)]
        for act in actions:
            if act():                                   # traded: refresh before the next decision
                positions = self.ex.get_positions()
                deltas = self.refresh_risk(positions)
        self.quote(positions, deltas, paused)

        for iid in self.instruments:
            self.drain_fills(iid)

        if now - self.last_book_log >= C.LOG_BOOKS_EVERY_S:
            self.last_book_log = now
            for iid, (b, bv, a, av) in self.tobs.items():
                self.log.row("books", [now, iid, b, bv, a, av, self.fair.get(iid), self.theo.get(iid)])

    def pnl(self):
        try:
            return self.ex.get_pnl(valuations=self.fair) if self.fair else self.ex.get_pnl()
        except Exception:
            return None

    def report(self):
        pnl = self.pnl()
        if pnl is not None and self.start_pnl is None:
            self.start_pnl = pnl
        self.session = pnl - self.start_pnl if pnl is not None and self.start_pnl is not None else None
        el = time.time() - self.start_time
        positions = {k: v for k, v in self.ex.get_positions().items() if v}
        self.refresh_risk(self.ex.get_positions())
        deltas = {r: round(d, 1) for r, d in self.eff.items() if abs(d) > 0.05}
        b = self.brain
        fmt = lambda x, s="+.2f": "n/a" if x is None else format(x, s)  # noqa: E731

        print(f"\n=== {time.strftime('%H:%M:%S')} | session #{b.s['sessions']} | "
              f"running {int(el) // 60}m{int(el) % 60:02d}s ===")
        print(f"PnL {fmt(pnl, '.2f')} | session {fmt(self.session)} | per min "
              f"{fmt(self.session / (el / 60) if self.session is not None and el > 60 else None)}")
        wr = 100.0 * self.mk_wins / self.mk_lots if self.mk_lots else None
        print(f"Fills {self.lots} lots, notional {self.notional:,.0f} | {C.MARKOUT_S:.0f}s markout "
              f"{fmt(self.mk_cash)} cash, win-rate {fmt(wr, '.0f')}%")
        print(f"Request budget {C.MAX_REQUESTS_PER_S}/s | skipped for budget: {self.throttled}")
        print(f"Positions {positions} | net exposure per root (ETF vs stocks netted) {deltas}")
        scored = sum(1 for n in b.s["names"].values() if n["n"] >= C.CONF_HALF_NAMES)
        alpha = ", ".join(f"{k}:{100 * b.alpha_scale(k):.0f}% (live {r.get('n_live', 0)}, hist {r.get('n_hist', 0)})"
                          for k, r in b.s["reg"].items())
        now = time.time()
        guarded = [i for i in self.instruments if self.guarded(i, now)]
        print(f"Jumps detected: {self.jumps} | quotes pulled now: {guarded or '-'} | clock pulls {self.blackouts}")
        ph = {i: f"{1000 * v:.0f}ms" for i, v in self.phase.items()}
        debts = {i: round(d, 1) for i, d in self.debt.items() if abs(d) >= 0.05}
        print(f"Clock phases: {ph or 'learning'} | lead-lag owed (ticks): {debts or '-'}")
        nw = b.s["news"]
        print(f"Learned: counterparties scored {scored}/{len(b.s['names'])} | news pause "
              f"{b.news_pause():.0f}s ({nw['n']} posts) | alpha {alpha or 'n/a'}")
        for iid in self.instruments:
            st = b.inst(iid)
            arm = self.epoch.get(iid, {}).get("arm", 1.0)
            extra = f" iv={st['iv']:.3f}" if st["iv"] else ""
            w = self.walls.get(iid)
            tick = self.instruments[iid].tick_size
            wall = f"wall={(w[1] - w[0]) / tick:.0f}t" if w else "wall=-"
            print(f"  {iid:<14} T{b.tier(iid)} {wall} vol1s={b.sigma_1s(iid):.4f} (conf {100 * b.vol_conf(iid):.0f}%) "
                  f"trust={'Y' if b.model_trusted(iid) else 'N'} fills={st['fills']} "
                  f"mkB={fmt(st['mk_bid'])} mkA={fmt(st['mk_ask'])} arm={arm}{extra}")
        return self.session

    def run(self):
        print(f"Instruments: {list(self.instruments)}")
        print(f"Roots: {dict(self.members)}")
        if self.index_def:
            print(f"Index definitions: {self.index_def}")
        print(f"Brain: session #{self.brain.s['sessions']}, data in {self.brain.path}")
        last_report = last_save = 0.0
        errors = 0
        try:
            while True:
                if not self.ex.is_connected():
                    print("Disconnected (teammate logged in?). Stopping.")
                    break
                try:
                    self.step()
                    errors = 0
                except Exception as e:
                    errors += 1
                    print(f"loop error {errors}/{C.MAX_CONSECUTIVE_ERRORS}: {e!r}")
                    if errors >= C.MAX_CONSECUTIVE_ERRORS:
                        print("Too many consecutive errors. Pulling orders and stopping.")
                        break
                now = time.time()
                if now - last_report >= C.REPORT_EVERY:
                    last_report = now
                    s = self.report()
                    if s is not None and s < C.MAX_LOSS:
                        print(f"Session PnL {s:.2f} < MAX_LOSS {C.MAX_LOSS}. Pulling orders and stopping.")
                        break
                if now - last_save >= C.SAVE_EVERY:
                    last_save = now
                    self.save()
                time.sleep(C.LOOP_SLEEP)
        except KeyboardInterrupt:
            print("Stopped by user.")
        finally:
            self.shutdown()

    def save(self):
        try:
            self.brain.save()
            self.log.flush()
        except OSError as e:
            print(f"save failed: {e}")

    def shutdown(self):
        if self.ex.is_connected():
            for iid in self.instruments:
                try:
                    while not self.rl.take():
                        time.sleep(0.05)
                    self.rl.pace()
                    self.ex.delete_orders(iid)
                except Exception as e:
                    print(f"[{iid}] delete_orders failed: {e}")
        el = (time.time() - self.start_time) / 60.0
        sess = self.session
        self.log.session_summary([time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.start_time)),
                                  round(el, 2), sess, self.lots, round(self.notional, 2),
                                  round(sess / el, 3) if sess is not None and el > 0 else None])
        self.save()
        self.log.close()


def main():
    if C.AUTO_PRETRAIN:
        try:
            from pretrain import pretrain
            pretrain()
        except Exception as e:                      # never block live trading on this
            print(f"Pretrain failed ({e!r}); continuing without it.")
    from optibook.synchronous_client import Exchange
    exchange = Exchange()
    exchange.connect()
    AdaptiveBot(exchange).run()


if __name__ == "__main__":
    main()
