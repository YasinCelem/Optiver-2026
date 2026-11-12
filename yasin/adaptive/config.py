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
    os.path.join(HERE, "..", "..", ".."),                     # local copy (moved to workspace root)
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
MAX_POS = 60                # our own cap per instrument (stocks with walls use SOFT_CAP_* below)
SOFT_CAP_NARROW, SOFT_CAP_WIDE = 40, 60   # passive fills / takes never push |position| above this
HARD_CAP = 80               # |position| reaches this -> bounded IOC down to half the soft cap
CUT_SLIP_TICKS = 2          # a hard-cap IOC may be priced this far through the wall
MAX_ROOT_DELTA = 80         # cap on net delta per underlying (in lots of the main instrument)
HEDGE_TRIGGER = 40          # roots with options: hedge delta when |delta| exceeds this
HEDGE_TARGET = 10           # ... down to this
MAX_LOSS = -100.0           # emergency stop on session PnL (valued at our fair); PnL here is in the tens
UNWIND_TRIGGER_FRAC = 0.75  # |exposure| above this share of the limit -> actively trade it down
UNWIND_MAX_COST_TICKS = 2.0 # ... but only if crossing costs at most this many ticks vs fair
UNWIND_CHUNK = 5            # lots per unwind trade
UNWIND_COOLDOWN_S = 2.0     # per root
REBALANCE_COST_TICKS = 1.0  # pay up to this per lot to flatten offsetting legs (e.g. +60 A / -60 B)
MAX_REQUESTS_PER_S = 25     # order inserts+deletes per second; the exchange force-disconnects above its limit
ACTION_GAP_S = 0.04         # and at least this long between two requests (evens out bursts; teammate's live setting)
MAX_CONSECUTIVE_ERRORS = 5# stop (and pull orders) after this many failed loops in a row
MAX_BOOK_AGE_S = 5.0        # ignore books older than this (None = off); auto-off if clocks disagree

# ---------------------------------------------------------------------------
# Quoting
# ---------------------------------------------------------------------------
QUOTE_VOLUME = 20          # the organizers' taker sends 20-lot orders
QUOTE_SIZES = {}            # per-stock override, e.g. {"AMD": 0} to switch passive quoting off
MIN_QUOTE_SIZE = 2          # don't bother quoting less than this
OPTION_QUOTE_VOLUME = 3

# Exchange "walls": large ladders re-posted around the true value once a second.
# Their midpoint is our fair value; we quote inside them, in front of other teams.
USE_WALLS = True
# Narrow stocks (AMD AVGO MU NVDA VRT): 3000/2000/1000 ladder, innermost >= WALL_BIG.
# Wide stocks (CRWV NBIS SNDK): 1000/750/500 ladder, innermost >= WALL_SMALL.
WALL_BIG, WALL_SMALL = 2500, 700
MAX_WALL_SPREAD_TICKS = 30  # wider than this is not a believable pair of walls
QUOTE_INSIDE = True         # step 1 tick in front of the best competing quote, while keeping our min edge
COMP_VOL = 5                # a competing level only counts once >= this many lots rest at/better than it (ignores 1-lot decoys)
# Minimum distance from the inventory-adjusted fair value, in ticks. Narrow stocks quote at the touch
# (safe because of the clock blackout); wide stocks stay >= 2 ticks away (edge there is 1.5-3 ticks/lot).
MIN_EDGE_NARROW_TICKS = 0.5
MIN_EDGE_WIDE_TICKS = 2.0
# Take resting quotes this far on the wrong side of our inventory-adjusted fair value
TAKE_EDGE_NARROW_TICKS, TAKE_EDGE_WIDE_TICKS = 1.0, 2.0
TAKE_COOLDOWN_S = 0.25

# Clock: the walls only move once per second, at a fixed phase per stock (verified on 4 h of data:
# 100% of moves in one 50 ms window). Fills within 100 ms after a move lose ~1.4 ticks/lot at 30 s.
# So: learn each stock's phase and rest NO quotes from CLOCK_PRE before to CLOCK_POST after it.
CLOCK_AWARE = True
CLOCK_PRE_S, CLOCK_POST_S = 0.12, 0.08
CLOCK_WIDE = False          # also black out wide stocks (their quotes are >= 2 ticks deep, so mostly safe)
CLOCK_MIN_SAMPLES, CLOCK_MIN_R = 8, 0.6   # how sure we must be of a phase before using it

# Lead-lag: these followers' walls copy their leader's move ~1-3 s later. follower: (leader, beta in
# follower ticks per leader tick), betas measured on the 4 h data (teammate used 0.22 / 0.95 / 0.40).
LEADS = {"NBIS": ("AVGO", 0.13), "CRWV": ("MU", 0.55), "SNDK": ("NVDA", 0.35)}
LL_LEAK_TICKS_S = 0.15      # an unpaid expected move fades by this many ticks per second
LL_TIMEOUT_S = 7.0          # a move the follower still hasn't made after this long is forgotten
LL_MIN_TICKS, LL_CAP_TICKS = 1.5, 30.0
LL_BIG_TICKS = 4.0          # expected move this large -> take with TAKE_SIZE_BIG

# Vol pause (replaces the strict 2-tick guard, which fired ~4x a minute and cost fills; the clock
# blackout now handles the normal 1-second moves): fair moves >= JUMP_TICKS within JUMP_WINDOW_S ->
# pull quotes on the stock AND its pair partners for JUMP_PAUSE_S.
JUMP_GUARD = True
JUMP_TICKS = 4.0
JUMP_WINDOW_S = 2.0
JUMP_PAUSE_S = 1.0
JUMP_CONFIRM_TICKS = 12     # a fair jump bigger than this must persist JUMP_CONFIRM_S before we trust it (restart)
JUMP_CONFIRM_S = 0.5
# After one of our quotes is filled, don't re-quote that side for this long
FILL_COOLDOWN_S = 1.5
FILL_COOLDOWN_TICKS = 0

# Correlated stocks (5 s return correlation, from the other-teams data). Positions in a
# pair offset each other in skew/limits at PAIR_NET_FACTOR x corr, and the partner can be
# used to unwind. Pairs whose stocks aren't tradable are ignored.
PAIRS = [("AVGO", "NBIS", 0.91), ("CRWV", "MU", 0.92), ("NVDA", "SNDK", 0.78)]
PAIR_NET_FACTOR = 0.7
EDGE_MIN_TICKS = 1
EDGE_VOL_MULT = 1.5         # edge >= EDGE_VOL_MULT * (1-second fair-value stdev)
COLD_SPREAD_FRAC = 0.5      # no walls + vol unknown: edge >= this * market half-spread (fades with vol confidence)
# Skew is measured in units of the current edge, so it scales with volatility.
# Curve: u + SKEW_CONVEXITY * u^3 (u = position / limit) -> gentle when small, hard near the limit.
SKEW_OWN = 0.4              # in ticks for stocks with walls: ~ (x + x^3) ticks in total at x = position / soft cap
SKEW_ROOT = 0.6
SKEW_CONVEXITY = 1.0
REQUOTE_TOL_TICKS = 1
MIN_REQUOTE_S = 0.4         # a quote that's merely less aggressive than wanted isn't moved more often than this

# ---------------------------------------------------------------------------
# Taking / arbitrage
# ---------------------------------------------------------------------------
ARB_MIN_TICKS = 1
BASKET_ARB = True           # ETF vs its constituent stocks
TAKE_MIN_TICKS = 3          # instruments without walls
TAKE_VOL_MULT = 3.0
MAX_TAKE_VOLUME = 20
TAKE_SIZE_BIG = 40          # when the lead-lag signal is large

# A model price (future carry, ETF NAV, ...) that sits further than this from
# the market, on average, is distrusted: we then fall back to the book.
DISTRUST_REL = 0.03

# ---------------------------------------------------------------------------
# Learning
# ---------------------------------------------------------------------------
VOL_SAMPLE_S = 5.0          # measure vol over 5s moves: book noise barely counts at that horizon
VOL_HALFLIFE_S = 120.0
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
# Confidence weighting (replaces on/off tiers). Each learned value acts in proportion
# to how much LIVE evidence supports it: weight = n / (n + HALF), so HALF samples = 50%.
# History (pretrain) is only a prior worth HIST_PRIOR_VOL_SAMPLES live vol samples, and
# loses that weight when live data disagrees with it.
# ---------------------------------------------------------------------------
HIST_PRIOR_VOL_SAMPLES = 24     # = 2 minutes of 5 s vol samples
HIST_DISAGREE_RATIO = 3.0       # live vol 3x away from history -> history weight 0
HIST_SAMPLE_WEIGHT = 0.2        # a replayed history sample counts 1/5 in the prediction model
CONF_HALF_VOL = 20              # live 5 s vol samples for 50% confidence
CONF_HALF_FILLS = 20            # live passive fills per side for 50% adverse-selection effect
CONF_HALF_ALPHA = 1000          # live labelled samples for 50% prediction-model effect
ALPHA_FULL_IMPROVEMENT = 0.05   # live error 5% below "no move" -> full prediction effect
CONF_HALF_NAMES = 30            # aggressive trades by a counterparty for 50% score weight

# Remaining hard thresholds (all on live data only)
TIER1_BOOKS = 300           # basis / smile maturity (display + derived instruments)
TIER2_FILLS = 40            # display only
TIER2_NEWS = 10             # news posts seen -> learn whether pausing is worth it
TIER3_FILLS = 150           # own fills -> bandit tunes the edge multiplier

# ---------------------------------------------------------------------------
# Loop / IO
# ---------------------------------------------------------------------------
LOOP_SLEEP = 0.05           # fast loop for the clock blackout; book reads are local. Order requests are paced separately.
REPORT_EVERY = 15.0
SAVE_EVERY = 30.0
LOG_BOOKS_EVERY_S = 1.0
