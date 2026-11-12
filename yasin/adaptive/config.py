"""
Settings for the adaptive bot. Check the challenge description and adjust the
risk numbers (POSITION_LIMIT especially) before running.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")          # brain.json + raw logs live here

# Historical data (context). Linked, not copied: the first folder that exists is
# used. Needs raw_pricebooks.csv.gz and raw_trades.csv.gz inside it.
HISTORY_DIRS = [
    os.path.join(HERE, "..", "..", "research_run_01"),        # Optibook workspace layout
    os.path.join(HERE, "..", "..", "..", "research run"),     # local copy
]
AUTO_PRETRAIN = True        # on start, replay any history the brain hasn't seen yet
HISTORY_TICK = None         # None = infer tick size per instrument from the prices

# ---------------------------------------------------------------------------
# Instruments
# ---------------------------------------------------------------------------
INSTRUMENTS = None          # None = everything tradable, or e.g. ["NVDA", "NVDA_DUAL"]
SKIP_IDS = set()            # instruments to never touch
TRADE_OPTIONS = True
LINKED_ROOTS = {}           # manual override {instrument_id: "root name"} if auto-grouping is wrong
OPTION_VOL_OVERRIDE = None  # force a Black-Scholes vol (e.g. 0.3); None = auto

# ---------------------------------------------------------------------------
# Risk
# ---------------------------------------------------------------------------
POSITION_LIMIT = 100        # exchange limit per instrument (check challenge!)
MAX_POS = 60                # our own cap per instrument
MAX_ROOT_DELTA = 80         # cap on net delta per underlying (in lots of the main instrument)
HEDGE_TRIGGER = 40          # roots with options: hedge delta when |delta| exceeds this
HEDGE_TARGET = 10           # ... down to this
MAX_LOSS = -3000.0          # emergency stop on session PnL (valued at our fair)
UNWIND_TRIGGER_FRAC = 0.75  # |exposure| above this share of the limit -> actively trade it down
UNWIND_MAX_COST_TICKS = 2.0 # ... but only if crossing costs at most this many ticks vs fair
UNWIND_CHUNK = 5            # lots per unwind trade
UNWIND_COOLDOWN_S = 2.0     # per root
REBALANCE_COST_TICKS = 1.0  # pay up to this per lot to flatten offsetting legs (e.g. +60 A / -60 B)
MAX_REQUESTS_PER_S = 20     # order inserts+deletes per second; the exchange force-disconnects above its limit
MAX_CONSECUTIVE_ERRORS = 5# stop (and pull orders) after this many failed loops in a row
MAX_BOOK_AGE_S = 5.0        # ignore books older than this (None = off); auto-off if clocks disagree

# ---------------------------------------------------------------------------
# Quoting
# ---------------------------------------------------------------------------
QUOTE_VOLUME = 5
OPTION_QUOTE_VOLUME = 3
EDGE_MIN_TICKS = 1
EDGE_VOL_MULT = 1.5         # edge >= EDGE_VOL_MULT * (1-second fair-value stdev)
COLD_SPREAD_FRAC = 0.5      # while vol is still unlearned: edge >= this * market half-spread
# Skew is measured in units of the current edge, so it scales with volatility.
# Curve: u + SKEW_CONVEXITY * u^3 (u = position / limit) -> gentle when small, hard near the limit.
SKEW_OWN = 0.5
SKEW_ROOT = 0.7
SKEW_CONVEXITY = 2.0
REQUOTE_TOL_TICKS = 1

# ---------------------------------------------------------------------------
# Taking / arbitrage
# ---------------------------------------------------------------------------
ARB_MIN_TICKS = 1
BASKET_ARB = True           # ETF vs its constituent stocks
TAKE_MIN_TICKS = 3
TAKE_VOL_MULT = 3.0
MAX_TAKE_VOLUME = 10

# A model price (future carry, ETF NAV, ...) that sits further than this from
# the market, on average, is distrusted: we then fall back to the book.
DISTRUST_REL = 0.03

# ---------------------------------------------------------------------------
# Learning
# ---------------------------------------------------------------------------
VOL_SAMPLE_S = 5.0          # measure vol over 5s moves: book noise barely counts at that horizon
VOL_HALFLIFE_S = 120.0
VOL_MIN_SAMPLES = 20        # vol counts as learned after this many 5s samples (and TIER1_BOOKS)
BASIS_HALFLIFE = 600        # samples; slow on purpose: learns persistent offsets only
MARKOUT_S = 5.0             # horizon to judge fills / flows
MARKOUT_TARGET_TICKS = 0.5  # markout we want per fill; below -> widen, above -> tighten
ALPHA_HORIZON_S = 5.0
ALPHA_SAMPLE_EVERY_S = 1.0
RIDGE_LAMBDA = 20.0
RIDGE_HALFLIFE_SAMPLES = 20000   # a sample this many samples old counts half (~40 min live with 8 stocks)
ALPHA_WEIGHT = 0.5          # how much of the predicted move we put in fair value
ALPHA_CLIP_TICKS = 3.0
BANDIT_ARMS = [0.7, 1.0, 1.4, 2.0]   # multipliers on the quoting edge
BANDIT_EPOCH_S = 30.0
NEWS_PAUSE_S = 3.0

# ---------------------------------------------------------------------------
# Unlock thresholds (counts persist across sessions in brain.json)
# ---------------------------------------------------------------------------
TIER1_BOOKS = 300           # book samples -> learned vol, basis correction, vol smile
TIER2_FILLS = 40            # own passive fills -> per-side adverse-selection edge
TIER2_NAME_TRADES = 30      # aggressive trades by a counterparty -> score them
TIER2_NEWS = 10             # news posts seen -> learn whether pausing is worth it
TIER3_FILLS = 150           # own fills -> bandit tunes the edge multiplier
TIER3_SAMPLES = 1500        # labelled samples -> short-term price prediction (only if it validates)

# ---------------------------------------------------------------------------
# Loop / IO
# ---------------------------------------------------------------------------
LOOP_SLEEP = 0.25           # keep >= 0.2 (rate limits)
REPORT_EVERY = 15.0
SAVE_EVERY = 30.0
LOG_BOOKS_EVERY_S = 1.0
