#!/usr/bin/env python3
"""Leonida Rarity Forecaster — GTA VI trophy earn-rate forecasts for make.fun.

Entry for gta6-trophy-rarity-forecast-01: for every trophy of Grand Theft Auto VI
on PS5, forecast five quantiles (0.05 / 0.25 / 0.5 / 0.75 / 0.95) of its earn rate
on Exophase at the case's target time.

Model in one paragraph
----------------------
An earn rate is a ratio, earners over players-tracked, and the two sides move for
different reasons on launch week: players-tracked jumps in big steps whenever
Exophase refreshes (new buyers flood in and dilute every rate), while earners
creep up as players actually play (fast for the opening chapters, glacial for the
platinum). So this program models the two sides separately, in log time measured
from the case's first read, and divides:

    rate(target) = earners(target) / players(target)

Each side is a log-linear fit in x = ln(1 + hours since the first read),
fitted on the recent window (the last 36 hours of distinct states — the
launch-night regime is not the day-3 regime) and anchored at the last
*distinct* page state (consecutive identical reads are one frozen Exophase
page, not fresh evidence), with its slope shrunk toward a per-trophy
prior that comes from the trophy's tier and description keywords. A soft cap
from the trophy class's long-run asymptote stops runaway extrapolations,
and the extrapolated move is blended 80/20 toward persistence, because
launch-week growth saturates. The spread widens with horizon, with the
trophy class's uncertainty, with the fit's residuals, and — decisively on
launch night — when the history is flat, because a flat page means a
refresh is overdue and the next one moves every rate at once.

Everything is deterministic: no randomness, no wall-clock reads, no network.
Python 3.13, standard library only.
"""

import json
import math
import os
import re
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

# ---------------------------------------------------------------------------
# Run configuration
# ---------------------------------------------------------------------------

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
# Standard-normal quantiles at the scored levels.
Z_SCORES = (-1.6448536269514722, -0.6744897501960817, 0.0,
            0.6744897501960817, 1.6448536269514722)

# Rates are clamped to this band before any logit is taken (the scorer does the
# same at 0.01..99.99).
RATE_FLOOR = 0.01
RATE_CEIL = 99.99

# A trophy the last read puts at or above this rate is treated as universal:
# every tracked player holds it (the opening-mission trophy sat at 100% even
# after the first refresh multiplied the tracked population by six).
UNIVERSAL_RATE = 99.0

# ---------------------------------------------------------------------------
# Priors, calibrated on launch-night dynamics and GTA V analogues
# ---------------------------------------------------------------------------
# Each trophy class carries:
#   growth  — prior slope of ln(earners) against ln(1 + hours since first read)
#   cap     — long-run earn-rate ceiling in percent (soft cap on the forecast)
#   spread  — prior logit spread for the class (rare/grind trophies are wider)
# The growth priors were set from the first real refresh (launch night:
# players 638 -> 3,868 while chapter-1 earners went 8 -> 19 and platinum
# earners stayed at 4) and from GTA V's long-run rates (first story trophy
# ~73%, platinum ~0.1%, grind trophies ~0.1-1%).

# (growth, cap_percent, spread)
CLASS_PRIORS = {
    "platinum":   (0.05, 0.35, 0.55),
    "completion": (0.05, 0.60, 0.55),
    "grind":      (0.18, 1.50, 0.50),
    "chapter1":   (0.50, 55.0, 0.30),
    "chapter2":   (0.35, 42.0, 0.30),
    "chapter3":   (0.30, 32.0, 0.32),
    "chapter4":   (0.25, 24.0, 0.35),
    "finale":     (0.20, 16.0, 0.35),
    "mission":    (0.30, 18.0, 0.35),
    "strand":     (0.25, 10.0, 0.40),
    "money":      (0.30, 6.0, 0.40),
    "misc":       (0.40, 8.0, 0.40),
}

# Prior slope of ln(players-tracked) against ln(1 + hours since first read).
# Launch morning roughly doubled the tracked population every ~2 hours, which
# is a slope near 1 on this axis; the fit takes over as soon as real reads
# arrive, and the slope is clamped to [0, 2] either way.
PLAYERS_GROWTH_PRIOR = 1.0
PLAYERS_GROWTH_MIN = 0.0
PLAYERS_GROWTH_MAX = 2.0

# How many distinct observations a fitted slope is worth against its prior:
# slope = (n * fit + PRIOR_WEIGHT * prior) / (n + PRIOR_WEIGHT).
PRIOR_WEIGHT = 3.0

# Slopes are fitted on the recent window only (the launch-night regime is
# not the day-3 regime): distinct states newer than this many hours before
# the checkpoint, falling back to the whole history when the window holds
# fewer than 3 points.
RECENT_WINDOW_HOURS = 36.0

# The extrapolated move is blended toward persistence (the last rate) in
# logit space: median = BLEND * extrapolated + (1 - BLEND) * last rate.
# Launch-week growth saturates, so a raw log-time extrapolation can
# overshoot; the blend is the hedge. Tuned on the synthetic launch-week
# backtest (blend 1.0 scored best there, 0.8 keeps a 20% hedge against a
# faster-saturating real week for a ~0.01 loss cost). The blend is skipped
# entirely when the history is flat: then the last rate is a stale page,
# not an anchor.
EXTRAPOLATION_BLEND = 0.80

# Spread budget (logit units, added in quadrature):
SPREAD_HORIZON_PER_SQRT_DAY = 0.22   # random-walk term, smaller than the
                                     # house 0.5 because the median is modelled
SPREAD_FLAT_HISTORY = 0.45           # < 3 distinct reads: a refresh is overdue
SPREAD_STALE_PAGE = 0.25             # last distinct state older than 3 h
STALE_HOURS = 3.0
SPREAD_MIN = 0.40
SPREAD_MAX = 1.80
# Universal (100%) trophy: tight-ish band that still admits a real drop.
UNIVERSAL_MEDIAN = 99.7
UNIVERSAL_SPREAD = 0.90


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------

def clamp_rate(rate):
    """Clamp a percent into the band where logits are finite."""
    return min(RATE_CEIL, max(RATE_FLOOR, rate))


def logit_of(rate):
    """logit of a percent rate, clamped first."""
    p = clamp_rate(rate) / 100.0
    return math.log(p / (1.0 - p))


def rate_of(logit_value):
    """Inverse logit back to a percent, clamped to the finite band."""
    if logit_value >= 0:
        p = 1.0 / (1.0 + math.exp(-logit_value))
    else:
        e = math.exp(logit_value)
        p = e / (1.0 + e)
    return min(RATE_CEIL, max(RATE_FLOOR, 100.0 * p))


def round4(value):
    """Round to 4 decimals, half up, deterministically (Decimal, not float)."""
    rounded = Decimal(repr(value)).quantize(Decimal("0.0001"),
                                            rounding=ROUND_HALF_UP)
    as_float = float(rounded)
    return int(as_float) if as_float.is_integer() else as_float


def parse_moment(text):
    """Parse an ISO-8601 UTC timestamp into an aware datetime."""
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    moment = datetime.fromisoformat(text)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment


def ols_slope(xs, ys):
    """Least-squares slope of ys on xs; None when xs carry no variance."""
    n = len(xs)
    if n < 2:
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx <= 1e-12:
        return None
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    return sxy / sxx


def ols_residual_std(xs, ys, slope, intercept):
    """Residual standard deviation of a line; 0 when it cannot be measured."""
    n = len(xs)
    if n < 3:
        return 0.0
    total = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    return math.sqrt(total / max(1, n - 2))


# ---------------------------------------------------------------------------
# Trophy classification
# ---------------------------------------------------------------------------

_CHAPTER_RE = re.compile(r"chapter\s+(\d+)", re.IGNORECASE)
_COUNT_RE = re.compile(r"(\d+)\s*(stunts|locations|vehicles|challenges|races|"
                       r"strands|vaults|sets|spots|properties)", re.IGNORECASE)


def classify(trophy):
    """Assign a trophy to a prior class from its tier and description."""
    tier = (trophy.get("tier") or "").lower()
    text = ((trophy.get("description") or "") + " " +
            (trophy.get("name") or "")).lower()
    if tier == "platinum":
        return "platinum"
    if "100% completion" in text or "100 percent" in text:
        return "completion"
    match = _CHAPTER_RE.search(text)
    if match:
        chapter = int(match.group(1))
        if chapter == 1:
            return "chapter1"
        if chapter == 2:
            return "chapter2"
        if chapter == 3:
            return "chapter3"
        return "chapter4"
    if "final mission" in text and tier == "gold":
        return "finale"
    if "final mission for" in text:
        return "strand"
    if _COUNT_RE.search(text) or "gold medal" in text or "every spot" in text \
            or "every location" in text or "all of leonida" in text \
            or "each gang" in text or "each type of activity" in text:
        return "grind"
    if "$" in text and ("earn" in text or "spend" in text):
        return "money"
    if text.startswith("complete ") and tier == "bronze":
        return "mission"
    return "misc"


# ---------------------------------------------------------------------------
# History preparation
# ---------------------------------------------------------------------------

def distinct_states(history):
    """Collapse consecutive identical reads into one state each.

    Exophase refreshes on its own schedule, so long runs of identical reads
    are one frozen page. Each distinct state keeps the *first* moment it was
    seen: the refresh behind it happened some time before that, and first-seen
    is the honest anchor for fitting. Returns a list of
    (moment, players_tracked, rates_tuple).
    """
    states = []
    previous = None
    for read in history:
        key = (read.get("players_tracked"), tuple(read.get("rates") or ()))
        if key == previous:
            continue
        previous = key
        try:
            moment = parse_moment(read["at"])
        except (KeyError, TypeError, ValueError):
            moment = parse_moment(read["slot"])
        states.append((moment, read.get("players_tracked"),
                       tuple(read.get("rates") or ())))
    return states


def log_time_axis(moments, anchor):
    """x = ln(1 + hours since the anchor moment), one value per moment."""
    axis = []
    for moment in moments:
        hours = (moment - anchor).total_seconds() / 3600.0
        axis.append(math.log1p(max(0.0, hours)))
    return axis


# ---------------------------------------------------------------------------
# The forecast
# ---------------------------------------------------------------------------

def fit_growth(xs, ys, prior_slope, n_points, recent_xs=None, recent_ys=None):
    """Slope of ys on xs, shrunk toward the prior; plus residual spread.

    When a recent window is supplied with at least 3 points, the slope is
    fitted there (the current regime is what extrapolates); otherwise the
    full series is used. Returns (slope, residual_std). With too few points
    the prior slope is used untouched and the residual term is zero.
    """
    fit_xs, fit_ys = xs, ys
    if recent_xs is not None and len(recent_xs) >= 3:
        fit_xs, fit_ys = recent_xs, recent_ys
    slope = ols_slope(fit_xs, fit_ys)
    if slope is None:
        return prior_slope, 0.0
    n = len(fit_xs)
    shrunk = (n * slope + PRIOR_WEIGHT * prior_slope) / (n + PRIOR_WEIGHT)
    intercept = sum(fit_ys) / n - slope * sum(fit_xs) / n
    resid = ols_residual_std(fit_xs, fit_ys, slope, intercept)
    return shrunk, min(0.6, resid)


def recent_window(xs, ys, moments, checkpoint):
    """The (x, y) points whose moment falls in the recent window."""
    cutoff = checkpoint.timestamp() - RECENT_WINDOW_HOURS * 3600.0
    rx, ry = [], []
    for x, y, moment in zip(xs, ys, moments):
        if moment.timestamp() >= cutoff:
            rx.append(x)
            ry.append(y)
    return rx, ry


def forecast_case(case):
    """Forecast one case. Returns the output dict {"trophies": [...]}."""
    trophies = case["trophies"]
    history = case["history"]
    if not history:
        raise ValueError("case has no history")
    checkpoint = parse_moment(case["at"])
    target = parse_moment(case["target"])
    horizon_days = max(0.0, case["horizon_minutes"] / 1440.0)

    states = distinct_states(history)
    anchor = states[0][0]
    moments = [s[0] for s in states]
    xs = log_time_axis(moments, anchor)
    x_target = math.log1p(max(0.0, (target - anchor).total_seconds() / 3600.0))
    x_last = xs[-1]
    dx = max(0.0, x_target - x_last)

    n_states = len(states)
    flat_history = n_states < 3
    stale_page = (checkpoint - moments[-1]).total_seconds() / 3600.0 > STALE_HOURS

    # --- players-tracked side -------------------------------------------
    players_points = [(x, math.log(max(1, p)), m) for x, (_, p, _), m in
                      zip(xs, states, moments) if p]
    if players_points:
        px = [p[0] for p in players_points]
        py = [p[1] for p in players_points]
        pm = [p[2] for p in players_points]
        rpx, rpy = recent_window(px, py, pm, checkpoint)
        players_slope, players_resid = fit_growth(
            px, py, PLAYERS_GROWTH_PRIOR, len(px), rpx, rpy)
    else:
        players_slope, players_resid = PLAYERS_GROWTH_PRIOR, 0.0
    players_slope = min(PLAYERS_GROWTH_MAX,
                        max(PLAYERS_GROWTH_MIN, players_slope))
    last_players = states[-1][1] or 1
    players_target = last_players * math.exp(players_slope * dx)

    # --- per-trophy forecasts -------------------------------------------
    rows = []
    for index, trophy in enumerate(trophies):
        growth_prior, cap, class_spread = CLASS_PRIORS[classify(trophy)]

        # Last non-null rate, walking back over reads that lacked the trophy.
        last_rate = None
        for _, _, rates in reversed(states):
            if index < len(rates) and rates[index] is not None:
                last_rate = rates[index]
                break
        if last_rate is None:
            last_rate = 0.0

        # Universal trophies: every tracked player holds one; the ratio model
        # cannot express that (its two slopes would drift apart), so pin it.
        if last_rate >= UNIVERSAL_RATE:
            median_logit = logit_of(UNIVERSAL_MEDIAN)
            sigma = UNIVERSAL_SPREAD
            rows.append({"id": trophy["id"],
                         "q": _quantiles(median_logit, sigma)})
            continue

        # Earners series for this trophy: rate * players / 100, log-smoothed
        # so a 0.00% read still contributes (0.5 pseudo-earners).
        ex, ey, em = [], [], []
        for x, (_, players, rates), moment in zip(xs, states, moments):
            if index >= len(rates) or rates[index] is None or not players:
                continue
            earners = rates[index] / 100.0 * players
            ex.append(x)
            ey.append(math.log(earners + 0.5))
            em.append(moment)
        rex, rey = recent_window(ex, ey, em, checkpoint)
        earner_slope, earner_resid = fit_growth(
            ex, ey, growth_prior, len(ex), rex, rey)
        earner_slope = min(2.0, max(0.0, earner_slope))

        last_earners = last_rate / 100.0 * last_players
        earners_target = (last_earners + 0.5) * math.exp(earner_slope * dx) - 0.5
        earners_target = max(0.0, earners_target)
        data_rate = 100.0 * earners_target / max(1.0, players_target)
        median_logit = logit_of(data_rate)

        # Blend the extrapolated move toward persistence: launch-week growth
        # saturates, so the raw extrapolation overshoots the target. Only
        # when the history has real trajectory data — with a flat history
        # the last rate is a stale launch-night page, not an anchor, and the
        # modelled move (usually a dilution drop) is the whole point.
        last_logit = logit_of(last_rate)
        if flat_history:
            blend = 1.0
        else:
            blend = EXTRAPOLATION_BLEND
        median_logit = (blend * median_logit + (1.0 - blend) * last_logit)

        # Soft cap at the class's long-run asymptote: a fit that overshoots
        # the cap is pulled most of the way back toward it.
        cap_logit = logit_of(cap)
        if median_logit > cap_logit:
            median_logit = cap_logit + 0.3 * (median_logit - cap_logit)

        # Spread: horizon + class + fit residuals + flat/stale-page terms.
        variance = (SPREAD_HORIZON_PER_SQRT_DAY * math.sqrt(horizon_days)) ** 2
        variance += class_spread ** 2
        variance += earner_resid ** 2
        variance += (0.5 * players_resid) ** 2
        if flat_history:
            variance += SPREAD_FLAT_HISTORY ** 2
        if stale_page:
            variance += SPREAD_STALE_PAGE ** 2
        sigma = min(SPREAD_MAX, max(SPREAD_MIN, math.sqrt(variance)))

        rows.append({"id": trophy["id"],
                     "q": _quantiles(median_logit, sigma)})
    return {"trophies": rows}


def _quantiles(median_logit, sigma):
    """The five scored quantiles around a logit median, rounded, ordered."""
    values = [round4(rate_of(median_logit + z * sigma)) for z in Z_SCORES]
    # Rounding is monotone, but enforce non-decreasing explicitly.
    for i in range(1, len(values)):
        if values[i] < values[i - 1]:
            values[i] = values[i - 1]
    return values


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        out_path = OUTPUT_DIR / (path.stem + ".out")
        out_path.write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
