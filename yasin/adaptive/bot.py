"""
Adaptive market maker: starts with pricing models that need no data and
unlocks data-driven features as its saved data grows.

TIER 0  (always on, from the first second; no data needed)
  - Dual listings: one shared fair value (liquidity-weighted microprice).
  - Futures: fair = underlying * exp(r*T).
  - ETFs: fair = index(constituents, divisor) * multiplier + cash component.
  - Options: Black-Scholes, using index_volatility if given, else market implied vol.
  - Cross-instrument arbitrage, IOC takes on big mispricings, delta hedging.
  - Conservative quoting while vol is unknown (edge >= half the market half-spread).
  - Pause quoting on news posts.
  - Model sanity check: if a model price is persistently >3% off the market,
    fall back to the book for that instrument.

TIER 1  (>= TIER1_BOOKS book samples per instrument)
  - Learned volatility sets the quoting edge (no longer tied to the spread).
  - Learned basis: persistent market-vs-model offset gets corrected.
  - Options: per-strike implied vols, fitted to a quadratic smile.

TIER 2  (>= TIER2_FILLS passive fills per instrument)
  - Adverse selection: each side widens or tightens by its own 5s markout.
  - Counterparty scoring: aggressors whose trades predict moves become a signal.
  - News: learns whether posts move prices; stops pausing if they don't.

TIER 3  (>= TIER3_FILLS fills / TIER3_SAMPLES samples)
  - Bandit chooses the edge multiplier per instrument by realised markout PnL.
  - Ridge regression predicts the 5s mid move from imbalance, trade flow,
    model deviation and informed flow; it is only used when it beats
    "predict zero" out of sample.

Everything learned is saved in data/brain.json. Raw logs go to data/raw/<session>/.
Every session appends a line to data/sessions.csv, so you can see whether it improves.

Run:  python yasin/adaptive/bot.py         Stop: Ctrl+C (orders are pulled).
Reset what it learned: delete yasin/adaptive/data/brain.json
"""
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
    def read_market(self, now, dt):
        self.tobs, self.mid = {}, {}
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
                self.mid[iid] = 0.5 * (tob[0] + tob[2])

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
                    micro = microprice(self.tobs[iid])
                    self.brain.update_basis(iid, micro - theo, abs(micro - theo) / max(abs(theo), 1e-9))
                if self.brain.model_trusted(iid):
                    self.fair_pre[iid] = theo + self.brain.basis(iid)
                elif iid in self.tobs:
                    self.fair_pre[iid] = microprice(self.tobs[iid])
                    self.warn(f"distrust-{iid}", f"[{iid}] model off by "
                              f"{100 * self.brain.inst(iid)['rel_err']:.1f}% -> using its own book")

        self._build_surfaces()
        for iid in self.instruments:
            if self.kind[iid] == "option":
                self._price_option(iid)

        # short-term alpha on top of the model (only once it validates)
        for iid, f in self.fair_pre.items():
            x = self._features(iid, f)
            self.features[iid] = x
            pred = self.brain.predict(type_name(self.instruments[iid]), x) if x else 0.0
            self.fair[iid] = f + C.ALPHA_WEIGHT * pred * self.instruments[iid].tick_size

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
                num += w * (microprice(tob) - b) / a
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
        self.eff = eff
        return raw

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
        if side == "bid":
            r, dchg = min(C.MAX_POS - pos, C.POSITION_LIMIT - pos), self.d1.get(iid, 0.0)
        else:
            r, dchg = min(C.MAX_POS + pos, C.POSITION_LIMIT + pos), -self.d1.get(iid, 0.0)
        if use_delta and abs(dchg) > 1e-9:
            rd = deltas.get(self.root[iid], 0.0)
            lim = (C.MAX_ROOT_DELTA - rd) / dchg if dchg > 0 else (C.MAX_ROOT_DELTA + rd) / -dchg
            r = min(r, math.floor(lim + 1e-9))
        return max(0, int(r))

    def ioc(self, iid, side, price, volume, wait=False):
        volume = int(volume)
        if volume <= 0:
            return 0
        if not self.rl.take():
            self.throttled += 1
            return 0
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
            for m in mem:
                dl = self.d1.get(m, 0.0)
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
        for iid, inst in self.instruments.items():
            if iid not in self.tobs or iid not in self.fair or not self.brain.model_trusted(iid):
                continue
            bid, bv, ask, av = self.tobs[iid]
            f, tick = self.fair[iid], inst.tick_size
            thresh = max(C.TAKE_MIN_TICKS * tick, C.TAKE_VOL_MULT * self.brain.sigma_1s(iid))
            if not self.brain.vol_learned(iid):
                thresh = max(thresh, ask - bid)        # cold start: only obvious mispricings
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

    def target_quotes(self, iid, positions, deltas):
        inst = self.instruments[iid]
        tick = inst.tick_size
        best_bid, _, best_ask, _ = self.tobs[iid]
        f = self.fair[iid]

        edge = max(C.EDGE_MIN_TICKS * tick, C.EDGE_VOL_MULT * self.brain.sigma_1s(iid))
        if not self.brain.vol_learned(iid):
            edge = max(edge, C.COLD_SPREAD_FRAC * (best_ask - best_bid) / 2)
        edge *= self.epoch[iid]["arm"] if iid in self.epoch else 1.0
        e_bid = max(C.EDGE_MIN_TICKS * tick, edge + self.brain.adverse_extra_ticks(iid, "bid") * tick)
        e_ask = max(C.EDGE_MIN_TICKS * tick, edge + self.brain.adverse_extra_ticks(iid, "ask") * tick)

        pos = positions.get(iid, 0)
        exposure = self.eff.get(self.root[iid], 0.0)
        dl = clip(self.d1.get(iid, 1.0), -1.0, 1.0)
        unit = max(tick, edge)
        skew = -unit * (C.SKEW_OWN * self._curve(pos / C.MAX_POS)
                        + C.SKEW_ROOT * self._curve(exposure / C.MAX_ROOT_DELTA) * dl)

        bid = min(round_down(f - e_bid + skew, tick), round_down(best_ask - tick, tick))
        ask = max(round_up(f + e_ask + skew, tick), round_up(best_bid + tick, tick))
        if bid >= ask:
            return None
        size = C.OPTION_QUOTE_VOLUME if self.kind[iid] == "option" else C.QUOTE_VOLUME
        vb = min(size, self.room(iid, "bid", positions, deltas, include_open=False))
        va = min(size, self.room(iid, "ask", positions, deltas, include_open=False))
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
        # reserve the whole replace up front: never delete without being able to re-insert
        if not self.rl.take(len(mine) + (1 if volume > 0 else 0)):
            self.throttled += 1
            return
        for o in mine:
            resp = self.ex.delete_order(iid, order_id=o.order_id)
            if resp is not None and not getattr(resp, "success", True):
                self.warn(f"del-{iid}-{side}", f"[{iid}] delete {side} {o.order_id} failed: "
                          f"{getattr(resp, 'error_reason', '?')} -> not re-quoting this loop")
                return
        if mine and any(o.side == side for o in self.ex.get_outstanding_orders(iid).values()):
            # Old order still live (failed or racing cancel): never stack a second one.
            return
        if volume > 0:
            resp = self.ex.insert_order(iid, price=price, volume=int(volume), side=side, order_type="limit")
            if not resp.success:
                self.warn(f"lim-{iid}-{side}", f"[{iid}] {side} {volume}@{price} rejected: {resp.error_reason}")

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
                if not paused and iid in self.tobs and iid in self.fair:
                    q = self.target_quotes(iid, positions, deltas)
                if q is None:
                    if outstanding and self.rl.take():
                        self.ex.delete_orders(iid)
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
        actions += [lambda: self.hedge(positions, deltas), lambda: self.unwind(positions, deltas, now)]
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
        scored = sum(1 for n in b.s["names"].values() if n["n"] >= C.TIER2_NAME_TRADES)
        alpha = ", ".join(f"{k}:{'ON' if b.alpha_active(k) else 'off'}({r['n']})" for k, r in b.s["reg"].items())
        nw = b.s["news"]
        print(f"Learned: counterparties scored {scored}/{len(b.s['names'])} | news pause "
              f"{b.news_pause():.0f}s ({nw['n']} posts) | alpha {alpha or 'n/a'}")
        for iid in self.instruments:
            st = b.inst(iid)
            arm = self.epoch.get(iid, {}).get("arm", 1.0)
            extra = f" iv={st['iv']:.3f}" if st["iv"] else ""
            print(f"  {iid:<14} T{b.tier(iid)} vol1s={b.sigma_1s(iid):.4f} basis={b.basis(iid):+.4f} "
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
