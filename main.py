#!/usr/bin/env python3
"""Forecast GTA VI trophy earn rates on Exophase at the case target.

Reads every <id>.in in the input dir ($MAKEFUN_INPUT_DIR, else /input) and
writes <id>.out to the output dir ($MAKEFUN_OUTPUT_DIR, else /output).

Method (logit space throughout): per trophy, a weighted least-squares trend
of logit(rate) against ln(1 + hours since launch) over the last 24 h of
reads, recency-weighted with a 12 h half-life; the slope is shrunk toward
the cross-trophy median slope. The median is capped at +-2.5 logit units
from the last rate, story-chain medians are forced non-increasing, and the
five quantiles come from a two-component normal mixture (75% narrow,
25% wide with sd 1.20 logit) so the tails stay honest about the chance of
a discrete Exophase refresh jump. Stdlib only, deterministic.
"""

import json
import math
import os
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP

LAUNCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)
LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
K_SHRINK = 1.0            # slope shrinkage pseudo-count
MAX_MOVE = 2.5            # max median move from last rate, logit units
RW_SIGMA = 0.35           # random-walk sigma per sqrt(day)
TAIL_SD = 1.20            # wide mixture component sd, logit
TAIL_W = 0.25             # wide mixture component weight
SPREAD_FLOOR = 0.20
SPREAD_CAP = 1.60
RESID_FLOOR = 0.05


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def clamp_rate(x):
    return min(99.99, max(0.01, x))


def logit(p):
    p = clamp_rate(p)
    return math.log(p / (100.0 - p))


def inv_logit(l):
    if l >= 0:
        return 100.0 / (1.0 + math.exp(-l))
    e = math.exp(l)
    return 100.0 * e / (1.0 + e)


def hours_since_launch(t):
    return (t - LAUNCH).total_seconds() / 3600.0


def x_of(t):
    return math.log1p(hours_since_launch(t))


def norm_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def round4_half_up(v):
    q = Decimal(repr(v)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
    if q == q.to_integral_value():
        return int(q)
    return float(q)


def median_of(values):
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0.0
    if n % 2 == 1:
        return s[n // 2]
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


def fit_trend(points):
    """Weighted least squares y = a + b*x. points: (x, y, w). Returns
    (a, b, resid_var, xbar, sxx, n) or None when there are < 3 points."""
    n = len(points)
    if n < 3:
        return None
    sw = sum(w for _, _, w in points)
    if sw <= 0:
        return None
    xbar = sum(w * x for x, _, w in points) / sw
    ybar = sum(w * y for _, y, w in points) / sw
    sxx = sum(w * (x - xbar) ** 2 for x, _, w in points)
    if sxx <= 0:
        return None
    sxy = sum(w * (x - xbar) * (y - ybar) for x, y, w in points)
    b = sxy / sxx
    a = ybar - b * xbar
    ssr = sum(w * (y - (a + b * x)) ** 2 for x, y, w in points)
    resid_var = ssr * n / (sw * (n - 2))
    return (a, b, resid_var, xbar, sxx, n)


def pool_adjacent_violators(values):
    """Force a sequence non-increasing by averaging rising adjacent groups."""
    groups = [[v] for v in values]
    i = 0
    while i < len(groups) - 1:
        mean_i = sum(groups[i]) / len(groups[i])
        mean_next = sum(groups[i + 1]) / len(groups[i + 1])
        if mean_i < mean_next - 1e-12:
            groups[i:i + 2] = [groups[i] + groups[i + 1]]
            if i > 0:
                i -= 1
        else:
            i += 1
    out = []
    for g in groups:
        m = sum(g) / len(g)
        out.extend([m] * len(g))
    return out


def forecast_case(case):
    trophies = case["trophies"]
    levels = case.get("levels") or list(LEVELS)
    target = parse_time(case["target"])
    x_target = x_of(target)
    horizon_days = case["horizon_minutes"] / 1440.0
    history = case.get("history") or []

    # Last known rate per trophy (latest non-null read, else 50).
    last = []
    for i in range(len(trophies)):
        rate = None
        for read in reversed(history):
            rates = read.get("rates") or []
            if i < len(rates) and rates[i] is not None:
                rate = float(rates[i])
                break
        last.append(rate if rate is not None else 50.0)

    # Per-trophy trend fit over the last 24 h of reads.
    fits = []
    if history:
        t_last = parse_time(history[-1]["at"])
        window_start = t_last.timestamp() - 24 * 3600
        for i in range(len(trophies)):
            points = []
            for read in history:
                t = parse_time(read["at"])
                if t.timestamp() < window_start:
                    continue
                rates = read.get("rates") or []
                if i >= len(rates) or rates[i] is None:
                    continue
                w = 0.5 ** ((t_last - t).total_seconds() / 43200.0)
                points.append((x_of(t), logit(float(rates[i])), w))
            fits.append(fit_trend(points))
    else:
        fits = [None] * len(trophies)

    slopes = [f[1] for f in fits if f is not None]
    b_med = median_of(slopes)

    medians = []   # median rate (percent) per trophy
    spreads = []   # s_center per trophy
    for i in range(len(trophies)):
        f = fits[i]
        last_logit = logit(last[i])
        if f is not None:
            a, b, resid_var, xbar, sxx, n = f
            lam = n / (n + K_SHRINK)
            b_star = lam * b + (1.0 - lam) * b_med
            med_logit = a + b_star * x_target
            med_logit = min(max(med_logit, last_logit - MAX_MOVE),
                            last_logit + MAX_MOVE)
            resid = math.sqrt(max(resid_var, RESID_FLOOR ** 2))
            var_floor = max(resid_var, RESID_FLOOR ** 2)
            if n > 2 and sxx > 0:
                se2 = var_floor * (1.0 / n + (x_target - xbar) ** 2 / sxx)
            else:
                se2 = 0.0
            s = math.sqrt(resid ** 2 + se2 +
                          (RW_SIGMA * math.sqrt(horizon_days)) ** 2)
        else:
            med_logit = last_logit
            s = RW_SIGMA * math.sqrt(horizon_days)
        s = min(max(s, SPREAD_FLOOR), SPREAD_CAP)
        med_rate = inv_logit(med_logit)
        # Edge cases on the rails.
        if last[i] >= 99.5:
            med_rate = min(med_rate, 95.0)
            s = max(s, 1.2)
        elif last[i] <= 0.05:
            med_rate = max(med_rate, 0.15)
            s = max(s, 0.8)
        medians.append(med_rate)
        spreads.append(s)

    # Story-chain monotonicity (medians only).
    chain_patterns = ["chapter 1", "chapter 2", "chapter 3", "chapter 4",
                      "final mission", "100% completion"]
    chain_idx = []
    seen = set()
    for pat in chain_patterns:
        for i, tr in enumerate(trophies):
            if i in seen:
                continue
            if pat in (tr.get("description") or "").lower():
                chain_idx.append(i)
                seen.add(i)
                break
    for i, tr in enumerate(trophies):
        if i not in seen and (tr.get("tier") or "").lower() == "platinum":
            chain_idx.append(i)
            seen.add(i)
            break
    if len(chain_idx) >= 2:
        chain_vals = [medians[i] for i in chain_idx]
        fixed = pool_adjacent_violators(chain_vals)
        for pos, i in enumerate(chain_idx):
            medians[i] = fixed[pos]

    # Quantiles from the two-component mixture, by bisection on its CDF.
    rows = []
    for i, tr in enumerate(trophies):
        m = logit(medians[i])
        s = spreads[i]
        lo = m - (6.0 * TAIL_SD + 1.0)
        hi = m + (6.0 * TAIL_SD + 1.0)
        qs = []
        for tau in levels:
            a_, b_ = lo, hi
            for _ in range(60):
                mid = 0.5 * (a_ + b_)
                cdf = ((1.0 - TAIL_W) * norm_cdf((mid - m) / s)
                       + TAIL_W * norm_cdf((mid - m) / TAIL_SD))
                if cdf < tau:
                    a_ = mid
                else:
                    b_ = mid
            q = inv_logit(0.5 * (a_ + b_))
            q = min(max(round4_half_up(q), 0), 100)
            if qs and q < qs[-1]:
                q = qs[-1]
            qs.append(q)
        rows.append({"id": tr["id"], "q": qs})
    return {"trophies": rows}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(in_dir) if n.endswith(".in"))
    for name in names:
        try:
            with open(os.path.join(in_dir, name), "r", encoding="utf-8") as fh:
                case = json.load(fh)
            result = forecast_case(case)
        except Exception:
            # Never crash: fall back to the last read's rates, flat; if the
            # case itself is unreadable, emit an empty (invalid but present)
            # forecast so the run still exits 0 with a file per input.
            try:
                rows = []
                history = case.get("history") or []
                for idx, tr in enumerate(case["trophies"]):
                    rate = 50.0
                    for read in reversed(history):
                        rates = read.get("rates") or []
                        if idx < len(rates) and rates[idx] is not None:
                            rate = float(rates[idx])
                            break
                    q = round4_half_up(rate)
                    rows.append({"id": tr["id"], "q": [q, q, q, q, q]})
                result = {"trophies": rows}
            except Exception:
                result = {"trophies": []}
        out_name = name[:-3] + ".out"
        with open(os.path.join(out_dir, out_name), "w",
                  encoding="utf-8") as fh:
            fh.write(json.dumps(result, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
