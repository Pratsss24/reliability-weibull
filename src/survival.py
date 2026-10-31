"""Non-parametric survival estimation, done without assuming a shape.

Everything here is distribution-free. That matters because the parametric
work in `weibull` has to be checked against something that does not share
its assumptions — a Weibull fit that disagrees with the Kaplan-Meier curve
is wrong about the data, not merely imprecise.

Four estimators:

`kaplan_meier`
    The survivor function. At each distance where a failure happened, the
    fraction of cars still at risk that got through it. Cars that are
    censored before that point simply leave the risk set: that is how a
    finisher contributes the information "survived this far" without
    contributing "failed here".
`nelson_aalen`
    The cumulative hazard, the sum of instantaneous failure rates. Easier
    to read than the survivor function when hazard is the quantity of
    interest, and the natural input to the Weibull diagnostic plot.
`logrank`
    Compares two groups without assuming either follows any particular
    law. At every event time it asks how many failures each group would
    have had if the groups were alike, and accumulates the discrepancy.
`cumulative_incidence`
    The Aalen-Johansen estimator, which is what a competing risk requires.
    One minus Kaplan-Meier silently answers "what fraction would fail if
    crashes were abolished", which is not a question anybody asked.

Variance for the survivor function uses Greenwood's formula, and the
confidence band is built on the log-log scale so that it cannot escape
[0, 1] — a symmetric band on S itself does, near the ends, which is where
these curves spend most of their time.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats


@dataclass
class Curve:
    """A step function of time, with a pointwise confidence band."""

    name: str
    times: np.ndarray
    values: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    at_risk: np.ndarray
    events: np.ndarray
    label: str = ""
    extras: dict = field(default_factory=dict)

    def at(self, t: float) -> float:
        """Value of the step function at t, right-continuous."""
        if self.times.size == 0 or t < self.times[0]:
            return float(self.values[0]) if self.values.size else 1.0
        index = int(np.searchsorted(self.times, t, side="right") - 1)
        return float(self.values[index])

    @property
    def total_events(self) -> int:
        return int(self.events.sum())


def _risk_table(time: np.ndarray, failed: np.ndarray
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Distinct event times, failures at each, and the risk set size.

    A censored observation at exactly an event time is conventionally
    treated as surviving that time, so it stays in the risk set for it.
    """
    time = np.asarray(time, dtype=float)
    failed = np.asarray(failed, dtype=bool)
    event_times = np.unique(time[failed])

    events = np.array([np.sum(failed & (time == t)) for t in event_times],
                      dtype=float)
    at_risk = np.array([np.sum(time >= t) for t in event_times], dtype=float)
    return event_times, events, at_risk


def kaplan_meier(time, failed, *, name: str = "", alpha: float = 0.05
                 ) -> Curve:
    """The product-limit survivor estimate with a log-log band."""
    times, events, at_risk = _risk_table(time, failed)
    if times.size == 0:
        empty = np.array([])
        return Curve(name=name, times=empty, values=empty, lower=empty,
                     upper=empty, at_risk=empty, events=empty)

    survival = np.cumprod(1.0 - events / at_risk)

    # Greenwood's variance of S, then transformed to the log-log scale.
    increments = events / (at_risk * np.maximum(at_risk - events, 1e-12))
    greenwood = np.cumsum(increments)
    z = stats.norm.ppf(1.0 - alpha / 2.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        log_s = np.log(np.maximum(survival, 1e-12))
        # Var(ln(-ln S)) = Greenwood / (ln S)^2
        # Capped because the last event can empty the risk set, which sends
        # Greenwood's variance to infinity. The band is already [0, 1] wide
        # there; the cap only keeps the exponential from overflowing.
        spread = np.minimum(z * np.sqrt(greenwood) / np.abs(log_s), 700.0)
        lower = survival ** np.exp(spread)
        upper = survival ** np.exp(-spread)

    lower = np.clip(np.nan_to_num(lower, nan=0.0), 0.0, 1.0)
    upper = np.clip(np.nan_to_num(upper, nan=1.0), 0.0, 1.0)

    return Curve(name=name, times=times, values=survival, lower=lower,
                 upper=upper, at_risk=at_risk, events=events,
                 label="survival",
                 extras={"greenwood": greenwood})


def nelson_aalen(time, failed, *, name: str = "", alpha: float = 0.05
                 ) -> Curve:
    """The cumulative hazard estimate."""
    times, events, at_risk = _risk_table(time, failed)
    if times.size == 0:
        empty = np.array([])
        return Curve(name=name, times=empty, values=empty, lower=empty,
                     upper=empty, at_risk=empty, events=empty)

    hazard = np.cumsum(events / at_risk)
    variance = np.cumsum(events / at_risk ** 2)
    z = stats.norm.ppf(1.0 - alpha / 2.0)
    spread = z * np.sqrt(variance)

    return Curve(name=name, times=times, values=hazard,
                 lower=np.maximum(hazard - spread, 0.0),
                 upper=hazard + spread, at_risk=at_risk, events=events,
                 label="cumulative hazard")


@dataclass(frozen=True)
class LogRank:
    """A two-group comparison that assumes no distribution."""

    first: str
    second: str
    observed_first: float
    expected_first: float
    statistic: float
    p_value: float
    n_first: int
    n_second: int

    @property
    def ratio(self) -> float:
        """Observed over expected failures in the first group."""
        if self.expected_first <= 0:
            return float("nan")
        return self.observed_first / self.expected_first


def logrank(time_a, failed_a, time_b, failed_b, *, first: str = "a",
            second: str = "b") -> LogRank:
    """Mantel-Haenszel log-rank test for two groups.

    At every distance where any car failed, the number of failures in group
    A is compared with the number expected under the hypergeometric that
    holds if the two groups share one survivor function.
    """
    time_a = np.asarray(time_a, dtype=float)
    failed_a = np.asarray(failed_a, dtype=bool)
    time_b = np.asarray(time_b, dtype=float)
    failed_b = np.asarray(failed_b, dtype=bool)

    pooled = np.unique(np.concatenate(
        [time_a[failed_a], time_b[failed_b]]))

    observed = expected = variance = 0.0
    for t in pooled:
        n_a = float(np.sum(time_a >= t))
        n_b = float(np.sum(time_b >= t))
        n = n_a + n_b
        if n <= 1:
            continue
        d_a = float(np.sum(failed_a & (time_a == t)))
        d = d_a + float(np.sum(failed_b & (time_b == t)))
        if d == 0:
            continue
        observed += d_a
        expected += d * n_a / n
        variance += (n_a * n_b * d * (n - d)) / (n * n * (n - 1.0))

    statistic = ((observed - expected) ** 2 / variance
                 if variance > 0 else 0.0)
    return LogRank(first=first, second=second, observed_first=observed,
                   expected_first=expected, statistic=float(statistic),
                   p_value=float(stats.chi2.sf(statistic, 1)),
                   n_first=int(time_a.size), n_second=int(time_b.size))


def cumulative_incidence(time, cause, target, *, name: str = "") -> Curve:
    """Aalen-Johansen cumulative incidence for one cause among several.

    `cause` is an integer array: 0 for censored, and a positive code for
    each competing cause. The estimator weights each cause-specific hazard
    increment by the probability of still being at risk at all, which is
    the all-cause survivor function:

        CIF_k(t) = sum over s <= t of  S(s-) * d_k(s) / n(s)

    One minus Kaplan-Meier, treating the other causes as censoring, gives a
    different and larger number. It answers a hypothetical question — what
    would fail if the other causes could not happen — and whether that
    question is the interesting one depends entirely on whether abolishing
    crashes is on the table.
    """
    time = np.asarray(time, dtype=float)
    cause = np.asarray(cause, dtype=int)
    any_event = cause > 0

    times = np.unique(time[any_event])
    if times.size == 0:
        empty = np.array([])
        return Curve(name=name, times=empty, values=empty, lower=empty,
                     upper=empty, at_risk=empty, events=empty)

    incidence = np.zeros(times.size)
    at_risk = np.zeros(times.size)
    events = np.zeros(times.size)
    overall = 1.0          # all-cause survival just before the current time
    running = 0.0

    for index, t in enumerate(times):
        n = float(np.sum(time >= t))
        d_all = float(np.sum(any_event & (time == t)))
        d_k = float(np.sum((cause == target) & (time == t)))
        at_risk[index] = n
        events[index] = d_k
        if n > 0:
            running += overall * d_k / n
            overall *= (1.0 - d_all / n)
        incidence[index] = running

    # A variance for the Aalen-Johansen estimator exists but is involved and
    # is not needed for the comparison this is used for, so no band is
    # claimed rather than a wrong one drawn.
    empty_band = np.full(times.size, np.nan)
    return Curve(name=name, times=times, values=incidence, lower=empty_band,
                 upper=empty_band, at_risk=at_risk, events=events,
                 label="cumulative incidence")


def naive_failure_probability(time, failed) -> Curve:
    """One minus Kaplan-Meier: the estimator a competing risk invalidates.

    Kept as a named function rather than written inline, because the point
    of the comparison is that this is what gets computed by default.
    """
    curve = kaplan_meier(time, failed)
    return Curve(name=curve.name, times=curve.times,
                 values=1.0 - curve.values, lower=1.0 - curve.upper,
                 upper=1.0 - curve.lower, at_risk=curve.at_risk,
                 events=curve.events, label="1 - Kaplan-Meier")


def median_survival(curve: Curve) -> float:
    """First distance at which survival drops to one half, if it does."""
    below = np.flatnonzero(curve.values <= 0.5)
    if below.size == 0:
        return float("nan")
    return float(curve.times[below[0]])


def restricted_mean(curve: Curve, horizon: float = 1.0) -> float:
    """Area under the survivor curve up to the horizon.

    The mean life is not estimable when the curve never reaches zero, which
    with 90 % censoring it does not. The restricted mean is the honest
    substitute: expected race fraction completed before a technical
    failure, counting a car that survives the race as completing all of it.
    """
    if curve.times.size == 0:
        return float(horizon)
    times = np.concatenate([[0.0], curve.times])
    values = np.concatenate([[1.0], curve.values])
    keep = times <= horizon
    times, values = times[keep], values[keep]
    widths = np.diff(np.concatenate([times, [horizon]]))
    return float(np.sum(values * widths))
