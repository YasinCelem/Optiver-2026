"""Pure pricing / math helpers. No exchange calls, no state."""
import math
from datetime import datetime, timezone

EPS = 1e-9


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _d1(S, K, T, r, sigma):
    return (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))


def bs_price(is_call, S, K, T, r, sigma):
    if S <= 0 or K <= 0:
        return None
    if T <= 0 or sigma <= 0:
        return max(0.0, S - K) if is_call else max(0.0, K - S)
    d1 = _d1(S, K, T, r, sigma)
    d2 = d1 - sigma * math.sqrt(T)
    disc = K * math.exp(-r * T)
    if is_call:
        return S * norm_cdf(d1) - disc * norm_cdf(d2)
    return disc * norm_cdf(-d2) - S * norm_cdf(-d1)


def bs_delta(is_call, S, K, T, r, sigma):
    if S <= 0 or K <= 0:
        return 0.0
    if T <= 0 or sigma <= 0:
        if is_call:
            return 1.0 if S > K else 0.0
        return -1.0 if S < K else 0.0
    d = norm_cdf(_d1(S, K, T, r, sigma))
    return d if is_call else d - 1.0


def implied_vol(is_call, price, S, K, T, r):
    """Bisection. None when the price carries no vol information."""
    if price is None or T <= 0 or S <= 0 or K <= 0:
        return None
    lower = max(0.0, S - K * math.exp(-r * T)) if is_call else max(0.0, K * math.exp(-r * T) - S)
    upper = S if is_call else K * math.exp(-r * T)
    if price <= lower + 1e-6 or price >= upper - 1e-6:
        return None
    lo, hi = 1e-4, 10.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if bs_price(is_call, S, K, T, r, mid) > price:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def solve_linear(A, b):
    """Gaussian elimination with partial pivoting. None if singular."""
    n = len(b)
    M = [list(map(float, A[i])) + [float(b[i])] for i in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda i: abs(M[i][c]))
        if abs(M[p][c]) < 1e-12:
            return None
        M[c], M[p] = M[p], M[c]
        for i in range(n):
            if i != c:
                f = M[i][c] / M[c][c]
                for j in range(c, n + 1):
                    M[i][j] -= f * M[c][j]
    return [M[i][n] / M[i][i] for i in range(n)]


def fit_quadratic(xs, ys):
    """Least squares y = c0 + c1 x + c2 x^2."""
    if len(xs) < 3:
        return None
    A = [[0.0] * 3 for _ in range(3)]
    b = [0.0] * 3
    for x, y in zip(xs, ys):
        v = (1.0, x, x * x)
        for i in range(3):
            b[i] += v[i] * y
            for j in range(3):
                A[i][j] += v[i] * v[j]
    return solve_linear(A, b)


def years_to_expiry(expiry):
    if expiry is None:
        return 0.0
    now = datetime.now(timezone.utc) if expiry.tzinfo else datetime.now()
    return max(0.0, (expiry - now).total_seconds() / (365.0 * 24 * 3600))


def tick_decimals(tick):
    s = f"{tick:.10f}".rstrip("0")
    return max(0, len(s.split(".")[1]) if "." in s else 0)


def round_down(price, tick):
    return round(math.floor(price / tick + EPS) * tick, tick_decimals(tick) + 1)


def round_up(price, tick):
    return round(math.ceil(price / tick - EPS) * tick, tick_decimals(tick) + 1)


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
        return 0.5 * (bid + ask)
    return (bid * avol + ask * bvol) / (bvol + avol)


def clip(x, lo, hi):
    return max(lo, min(hi, x))
