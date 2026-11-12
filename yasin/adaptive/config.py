"""
Settings for the adaptive bot. Check the challenge description and adjust the
risk numbers (POSITION_LIMIT especially) before running.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")          # brain.json + raw logs live here

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

# ---------------------------------------------------------------------------
# Quoting
# ---------------------------------------------------------------------------
QUOTE_VOLUME = 5
OPTION_QUOTE_VOLUME = 3
EDGE_MIN_TICKS = 1
EDGE_VOL_MULT = 1.5         # edge >= EDGE_VOL_MULT * (1-second fair-value stdev)
COLD_SPREAD_FRAC = 0.5      # while vol is still unlearned: edge >= this * market half-spread
SKEW_OWN_TICKS = 3
SKEW_ROOT_TICKS = 3
REQUOTE_TOL_TICKS = 1

# ---------------------------------------------------------------------------
# Taking / arbitrage
# ---------------------------------------------------------------------------
ARB_MIN_TICKS = 1
TAKE_MIN_TICKS = 3
TAKE_VOL_MULT = 3.0
MAX_TAKE_VOLUME = 10

# A model price (future carry, ETF NAV, ...) that sits further than this from
# the market, on average, is distrusted: we then fall back to the book.
DISTRUST_REL = 0.03

# ---------------------------------------------------------------------------
# Learning
# ---------------------------------------------------------------------------
VOL_HALFLIFE_S = 30.0
BASIS_HALFLIFE = 600        # samples; slow on purpose: learns persistent offsets only
MARKOUT_S = 5.0             # horizon to judge fills / flows
MARKOUT_TARGET_TICKS = 0.5  # markout we want per fill; below -> widen, above -> tighten
ALPHA_HORIZON_S = 5.0
ALPHA_SAMPLE_EVERY_S = 1.0
RIDGE_LAMBDA = 20.0
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
