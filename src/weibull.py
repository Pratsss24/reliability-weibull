"""Weibull fitting with right-censored data, and what happens without it.

The Weibull survivor function is

    S(t) = exp( -(t / eta) ** beta )

and the shape parameter is the whole reason reliability engineering uses
it rather than something simpler:

    beta < 1   the hazard falls with age: infant mortality, the failures
               that are built in and shake out early
    beta = 1   the hazard is constant: failures arrive at random, with no
               memory of how far the car has run (the exponential case)
    beta > 1   the hazard rises with age: wear-out, accumulated heat, load
               cycles, abrasion

So beta is not a curve-fitting nuisance. It is a claim about the physics,
and it is testable against the data.

**The censored likelihood.** Ninety per cent of the cars in this dataset
did not fail — they finished. Each one still carries information: "this car
ran this far without failing". The likelihood that uses it is

    log L = sum over failures [ ln beta - beta ln eta + (beta-1) ln t ]
          - sum over ALL observations [ (t / eta) ** beta ]

The first sum is over the 311 cars that failed; the second is over all
3,257, because a survivor contributes its survival probability rather than
a density. Dropping the survivors, or pretending they failed at the flag,
both give the wrong answer, and `naive_fits` measures how wrong.

**How it is solved.** For any fixed beta the scale has a closed form,

    eta(beta) = [ sum over all t ** beta / d ] ** (1 / beta)

with d the number of failures. Substituting it back leaves one equation in
beta alone,

    1/beta + (1/d) sum over failures ln t
           - [ sum over all t**beta ln t ] / [ sum over all t**beta ]  =  0

which is monotone decreasing in beta and so has exactly one root, found by
bisection rather than by a general optimiser that might wander. At
beta = 1 the scale equation collapses to eta = (total time on test) / d,
the textbook exponential estimator, which is what the tests check against.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize, stats


@dataclass(frozen=True)
class Weibull:
    """A fitted two-parameter Weibull."""

    name: str
    beta: float               # shape
    eta: float                # scale, in race fractions
    n: int
    failures: int
    log_likelihood: float
    beta_se: float = float("nan")
    eta_se: float = float("nan")
    method: str = "censored MLE"

    @property
    def aic(self) -> float:
        return 4.0 - 2.0 * self.log_likelihood

    @property
    def censoring_rate(self) -> float:
        return 1.0 - self.failures / self.n if self.n else float("nan")

    @property
    def regime(self) -> str:
        if self.beta_se == self.beta_se:          # not NaN
            low, high = self.beta_confidence()
            if high < 1.0:
                return "infant mortality"
            if low > 1.0:
                return "wear-out"
            return "indistinguishable from constant hazard"
        return "infant mortality" if self.beta < 1 else "wear-out"

    def beta_confidence(self, alpha: float = 0.05) -> tuple[float, float]:
        """Confidence interval for beta, on the log scale.

        beta is positive and its sampling distribution is skewed, so the
        interval is built for ln beta and transformed back; a symmetric
        interval on beta itself can reach below zero.
        """
        if not np.isfinite(self.beta_se) or self.beta_se <= 0:
            return (float("nan"), float("nan"))
        z = stats.norm.ppf(1.0 - alpha / 2.0)
        spread = z * self.beta_se / self.beta
        return (self.beta * np.exp(-spread), self.beta * np.exp(spread))

    def survival(self, t):
        t = np.asarray(t, dtype=float)
        return np.exp(-np.power(np.maximum(t, 0.0) / self.eta, self.beta))

    def hazard(self, t):
        t = np.maximum(np.asarray(t, dtype=float), 1e-12)
        return (self.beta / self.eta) * np.power(t / self.eta,
                                                 self.beta - 1.0)

    def cumulative_hazard(self, t):
        t = np.asarray(t, dtype=float)
        return np.power(np.maximum(t, 0.0) / self.eta, self.beta)

    def quantile(self, p: float) -> float:
        return float(self.eta * (-np.log1p(-p)) ** (1.0 / self.beta))

    @property
    def b10_life(self) -> float:
        """Distance by which a tenth of the population has failed."""
        return self.quantile(0.10)

    @property
    def mean_life(self) -> float:
        from scipy.special import gamma
        return float(self.eta * gamma(1.0 + 1.0 / self.beta))


def log_likelihood(time, failed, beta: float, eta: float) -> float:
    """Right-censored Weibull log-likelihood."""
    time = np.asarray(time, dtype=float)
    failed = np.asarray(failed, dtype=bool)
    if beta <= 0 or eta <= 0:
        return -np.inf
    scaled = np.power(time / eta, beta)
    event_term = np.sum(np.log(beta / eta)
                        + (beta - 1.0) * np.log(time[failed] / eta))
    return float(event_term - np.sum(scaled))


def _scale_for(time: np.ndarray, failures: int, beta: float) -> float:
    """eta given beta, in closed form."""
    return float((np.sum(np.power(time, beta)) / failures) ** (1.0 / beta))


def _shape_equation(beta: float, time: np.ndarray,
                    log_failed_sum: float, failures: int) -> float:
    powered = np.power(time, beta)
    weighted = np.sum(powered * np.log(time))
    total = np.sum(powered)
    return (1.0 / beta + log_failed_sum / failures - weighted / total)


def fit(time, failed, *, name: str = "", floor: float = 1e-6) -> Weibull:
    """Maximum likelihood with right censoring.

    `floor` keeps a time of exactly zero out of the logarithms. With the
    half-lap convention in `lifetimes` no observation should be at zero,
    but a caller passing raw data should not get a silent NaN.
    """
    time = np.maximum(np.asarray(time, dtype=float), floor)
    failed = np.asarray(failed, dtype=bool)
    failures = int(failed.sum())
    if failures < 2:
        raise ValueError("need at least two failures to fit a shape")

    log_failed_sum = float(np.sum(np.log(time[failed])))
    args = (time, log_failed_sum, failures)

    # The equation is monotone decreasing in beta, so bracket and bisect.
    low, high = 1e-3, 1.0
    while _shape_equation(high, *args) > 0 and high < 1e4:
        low, high = high, high * 2.0
    beta = float(optimize.brentq(_shape_equation, low, high, args=args,
                                 xtol=1e-12, rtol=1e-14))
    eta = _scale_for(time, failures, beta)

    beta_se, eta_se = _standard_errors(time, failed, beta, eta)
    return Weibull(name=name, beta=beta, eta=eta, n=int(time.size),
                   failures=failures,
                   log_likelihood=log_likelihood(time, failed, beta, eta),
                   beta_se=beta_se, eta_se=eta_se)


def _standard_errors(time, failed, beta: float, eta: float
                     ) -> tuple[float, float]:
    """From the observed information, by numerical second derivatives.

    The analytic Hessian is available but the numerical one is checked
    against the likelihood itself, which is the quantity that matters, and
    it cannot drift out of agreement with the fit when the model changes.
    """
    step_beta = max(beta * 1e-4, 1e-7)
    step_eta = max(eta * 1e-4, 1e-7)

    def value(b, e):
        return log_likelihood(time, failed, b, e)

    centre = value(beta, eta)
    d_bb = (value(beta + step_beta, eta) - 2 * centre
            + value(beta - step_beta, eta)) / step_beta ** 2
    d_ee = (value(beta, eta + step_eta) - 2 * centre
            + value(beta, eta - step_eta)) / step_eta ** 2
    d_be = ((value(beta + step_beta, eta + step_eta)
             - value(beta + step_beta, eta - step_eta)
             - value(beta - step_beta, eta + step_eta)
             + value(beta - step_beta, eta - step_eta))
            / (4 * step_beta * step_eta))

    information = -np.array([[d_bb, d_be], [d_be, d_ee]])
    try:
        covariance = np.linalg.inv(information)
    except np.linalg.LinAlgError:
        return float("nan"), float("nan")
    if covariance[0, 0] <= 0 or covariance[1, 1] <= 0:
        return float("nan"), float("nan")
    return float(np.sqrt(covariance[0, 0])), float(np.sqrt(covariance[1, 1]))


def fit_exponential(time, failed, *, name: str = "") -> Weibull:
    """The beta = 1 special case, in closed form.

    eta is the total time on test divided by the number of failures. No
    iteration, no optimiser — which makes it the anchor the general fitter
    is checked against.
    """
    time = np.asarray(time, dtype=float)
    failed = np.asarray(failed, dtype=bool)
    failures = int(failed.sum())
    if failures == 0:
        raise ValueError("no failures")
    eta = float(np.sum(time) / failures)
    return Weibull(name=name, beta=1.0, eta=eta, n=int(time.size),
                   failures=failures,
                   log_likelihood=log_likelihood(time, failed, 1.0, eta),
                   method="exponential, closed form")


def shape_test(time, failed) -> dict:
    """Is beta different from 1 by more than sampling noise?

    A likelihood ratio against the exponential, which is the Weibull with
    beta pinned at one. Two nested models, one degree of freedom.
    """
    full = fit(time, failed)
    reduced = fit_exponential(time, failed)
    statistic = 2.0 * (full.log_likelihood - reduced.log_likelihood)
    return {
        "beta": full.beta,
        "statistic": float(statistic),
        "p_value": float(stats.chi2.sf(max(statistic, 0.0), 1)),
        "log_likelihood_full": full.log_likelihood,
        "log_likelihood_exponential": reduced.log_likelihood,
    }


# -- what censoring is worth ----------------------------------------------

def naive_fits(time, failed, *, name: str = "") -> dict[str, Weibull]:
    """The correct fit beside the two ways of getting it wrong.

    `drop censored`
        Keep only the cars that failed and fit an ordinary uncensored
        Weibull. This is what happens when survivors are treated as
        missing data. Every long life is thrown away, so the scale
        collapses toward the failures' own range.
    `censored as failures`
        Count every finisher as having failed at the flag. The sample size
        is preserved and the physics is destroyed: a population in which
        90 % of units "fail" at almost exactly the same distance looks
        like a hard wear-out limit, and beta goes up accordingly.
    """
    time = np.asarray(time, dtype=float)
    failed = np.asarray(failed, dtype=bool)

    correct = fit(time, failed, name=f"{name} (censored MLE)")

    only_failures = time[failed]
    dropped = fit(only_failures, np.ones_like(only_failures, dtype=bool),
                  name=f"{name} (censored dropped)")

    as_failures = fit(time, np.ones_like(time, dtype=bool),
                      name=f"{name} (censored counted as failures)")

    return {"censored MLE": correct,
            "drop censored": dropped,
            "censored as failures": as_failures}


def probability_plot(curve, fitted: Weibull | None = None) -> dict:
    """The Weibull plot: ln(-ln S) against ln t should be a straight line.

    If the data really is Weibull, the transformed points fall on a line
    whose slope is beta. The R-squared of that line is a shape check that
    owes nothing to the fitted model, and the slope is an independent
    estimate of beta that the maximum likelihood fit has to agree with.
    """
    keep = (curve.values > 0) & (curve.values < 1) & (curve.times > 0)
    x = np.log(curve.times[keep])
    y = np.log(-np.log(curve.values[keep]))
    if x.size < 3:
        return {"slope": float("nan"), "intercept": float("nan"),
                "r_squared": float("nan"), "x": x, "y": y}

    slope, intercept, r_value, _, stderr = stats.linregress(x, y)
    out = {"slope": float(slope), "intercept": float(intercept),
           "r_squared": float(r_value ** 2), "slope_se": float(stderr),
           "eta_from_plot": float(np.exp(-intercept / slope)),
           "x": x, "y": y}
    if fitted is not None:
        out["beta_mle"] = fitted.beta
        out["beta_gap"] = abs(slope - fitted.beta)
    return out
