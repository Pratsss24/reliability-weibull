"""Tests anchored on closed forms and hand-computable examples.

Survival estimators are easy to write and easy to get subtly wrong, and a
wrong one produces a plausible curve rather than an error. So almost
nothing here compares against stored output. Instead:

* **Closed forms.** With the shape pinned at 1 the Weibull scale is the
  total time on test over the number of failures, with no iteration at all.
  That exact answer pins the general fitter. A Kaplan-Meier curve on
  uncensored data must equal one minus the empirical distribution
  function, exactly.
* **Hand-computable examples.** The six-observation Kaplan-Meier worked
  through below can be checked on paper, Greenwood's variance included.
* **Recovery of a known truth.** Simulated data from a Weibull whose
  parameters are chosen in advance, with censoring applied deliberately.
  The censored likelihood has to find them; the naive alternatives have to
  fail in the specific direction claimed in the write-up.
* **Internal consistency.** The cumulative incidence functions for all
  causes plus the all-cause survival must sum to exactly one at every
  time. A log-rank test of a group against itself must give zero.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import data          # noqa: E402
import lifetimes     # noqa: E402
import survival      # noqa: E402
import weibull       # noqa: E402


@pytest.fixture(scope="module")
def entries():
    return data.load()


@pytest.fixture(scope="module")
def lives(entries):
    return lifetimes.build(entries)


@pytest.fixture(scope="module")
def observed(lives):
    return lifetimes.arrays(lives)


# -- closed forms ---------------------------------------------------------

def test_the_exponential_scale_is_total_time_over_failures():
    """No iteration involved: this one can be done by hand."""
    time = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    failed = np.array([True, False, True, False, True])
    fit = weibull.fit_exponential(time, failed)
    assert fit.eta == pytest.approx(15.0 / 3.0)
    assert fit.beta == 1.0


def test_the_general_fitter_agrees_with_the_closed_form_when_beta_is_one():
    """Data generated with a constant hazard must come back with beta ~ 1.

    And when it does, the fitted scale must match the closed-form
    exponential estimator, because that is the same estimator.
    """
    rng = np.random.default_rng(4)
    sample = rng.exponential(3.0, 60_000)
    limit = 4.0
    time = np.minimum(sample, limit)
    failed = sample <= limit

    general = weibull.fit(time, failed)
    closed = weibull.fit_exponential(time, failed)
    assert general.beta == pytest.approx(1.0, abs=0.03)
    assert general.eta == pytest.approx(closed.eta, rel=0.03)
    assert general.eta == pytest.approx(3.0, rel=0.03)


def test_the_likelihood_is_maximised_at_the_fitted_parameters():
    """A fit that is not a maximum is not a maximum likelihood fit."""
    rng = np.random.default_rng(8)
    sample = 2.0 * rng.weibull(1.7, 5_000)
    time = np.minimum(sample, 2.5)
    failed = sample <= 2.5

    fit = weibull.fit(time, failed)
    best = weibull.log_likelihood(time, failed, fit.beta, fit.eta)
    for d_beta, d_eta in ((0.02, 0), (-0.02, 0), (0, 0.05), (0, -0.05),
                          (0.02, 0.05), (-0.02, -0.05)):
        nearby = weibull.log_likelihood(time, failed, fit.beta + d_beta,
                                        fit.eta + d_eta)
        assert nearby < best


def test_weibull_quantiles_invert_the_survivor_function():
    fit = weibull.Weibull(name="", beta=1.8, eta=4.0, n=10, failures=5,
                          log_likelihood=0.0)
    for p in (0.01, 0.1, 0.5, 0.9):
        t = fit.quantile(p)
        assert float(fit.survival(t)) == pytest.approx(1.0 - p, abs=1e-12)
    assert fit.b10_life == pytest.approx(fit.quantile(0.10))


def test_the_mean_life_matches_a_direct_simulation():
    fit = weibull.Weibull(name="", beta=2.3, eta=5.0, n=10, failures=5,
                          log_likelihood=0.0)
    rng = np.random.default_rng(21)
    drawn = 5.0 * rng.weibull(2.3, 500_000)
    assert fit.mean_life == pytest.approx(float(drawn.mean()), rel=0.01)


# -- Kaplan-Meier, by hand ------------------------------------------------

def test_kaplan_meier_on_an_example_that_can_be_checked_on_paper():
    """Six units: failures at 2, 5, 8; censored at 3, 6, 9.

    n = 6 at t = 2, one failure          S = 5/6            = 0.833333
    n = 4 at t = 5 (3 censored out), one S = 5/6 * 3/4      = 0.625
    n = 2 at t = 8 (6 censored out), one S = 5/6 * 3/4 * 1/2 = 0.3125
    """
    time = np.array([2.0, 3.0, 5.0, 6.0, 8.0, 9.0])
    failed = np.array([True, False, True, False, True, False])
    curve = survival.kaplan_meier(time, failed)

    assert list(curve.times) == [2.0, 5.0, 8.0]
    assert list(curve.at_risk) == [6.0, 4.0, 2.0]
    assert curve.values == pytest.approx([5 / 6, 0.625, 0.3125])
    assert curve.at(2.5) == pytest.approx(5 / 6)
    assert curve.at(1.0) == pytest.approx(5 / 6)   # before the first event
    assert curve.at(100.0) == pytest.approx(0.3125)


def test_greenwood_variance_on_the_same_example():
    """Sum of d / (n (n - d)) accumulated at each event time."""
    time = np.array([2.0, 3.0, 5.0, 6.0, 8.0, 9.0])
    failed = np.array([True, False, True, False, True, False])
    curve = survival.kaplan_meier(time, failed)

    expected = np.cumsum([1 / (6 * 5), 1 / (4 * 3), 1 / (2 * 1)])
    assert curve.extras["greenwood"] == pytest.approx(expected)


def test_kaplan_meier_without_censoring_is_one_minus_the_ecdf():
    rng = np.random.default_rng(2)
    sample = rng.exponential(1.0, 2_000)
    curve = survival.kaplan_meier(sample, np.ones_like(sample, dtype=bool))
    for t in (0.2, 0.5, 1.0, 2.0, 3.0):
        assert curve.at(t) == pytest.approx(float((sample > t).mean()),
                                            abs=1e-12)


def test_a_censored_observation_stays_in_the_risk_set_at_its_own_time():
    """The convention: censored at exactly an event time means it survived.

    Getting this backwards shifts every curve slightly, and nothing about
    the output looks wrong when it happens.
    """
    time = np.array([1.0, 1.0, 2.0])
    failed = np.array([True, False, True])
    curve = survival.kaplan_meier(time, failed)
    assert curve.at_risk[0] == 3.0
    assert curve.values[0] == pytest.approx(2 / 3)


def test_the_confidence_band_stays_inside_zero_and_one():
    """Why the band is built on the log-log scale."""
    rng = np.random.default_rng(6)
    sample = rng.exponential(1.0, 40)
    curve = survival.kaplan_meier(sample, np.ones_like(sample, dtype=bool))
    assert np.all(curve.lower >= 0.0)
    assert np.all(curve.upper <= 1.0)
    assert np.all(curve.lower <= curve.values + 1e-12)
    assert np.all(curve.upper >= curve.values - 1e-12)


def test_survival_and_cumulative_hazard_agree():
    """S = exp(-H) holds closely when no single risk set is tiny."""
    rng = np.random.default_rng(10)
    sample = rng.exponential(2.0, 5_000)
    limit = 3.0
    time = np.minimum(sample, limit)
    failed = sample <= limit

    km = survival.kaplan_meier(time, failed)
    na = survival.nelson_aalen(time, failed)
    for t in (0.5, 1.0, 2.0):
        assert km.at(t) == pytest.approx(float(np.exp(-na.at(t))), abs=0.01)


def test_restricted_mean_of_a_curve_that_never_fails_is_the_horizon():
    time = np.array([0.4, 0.8, 1.0])
    failed = np.array([False, False, False])
    curve = survival.kaplan_meier(time, failed)
    assert survival.restricted_mean(curve, 1.0) == pytest.approx(1.0)


def test_median_survival_is_undefined_when_the_curve_never_halves():
    time = np.array([1.0, 2.0, 3.0, 4.0])
    failed = np.array([True, False, False, False])
    curve = survival.kaplan_meier(time, failed)
    assert np.isnan(survival.median_survival(curve))


# -- the log-rank test ----------------------------------------------------

def test_a_group_against_itself_gives_no_evidence_of_difference():
    rng = np.random.default_rng(12)
    sample = rng.exponential(1.0, 300)
    failed = rng.random(300) < 0.6
    result = survival.logrank(sample, failed, sample, failed)
    assert result.statistic == pytest.approx(0.0, abs=1e-9)
    assert result.p_value == pytest.approx(1.0)
    assert result.ratio == pytest.approx(1.0, abs=1e-9)


def test_the_log_rank_test_finds_a_difference_that_is_really_there():
    rng = np.random.default_rng(14)
    fragile = rng.exponential(1.0, 400)
    sturdy = rng.exponential(3.0, 400)
    result = survival.logrank(fragile, np.ones(400, dtype=bool),
                              sturdy, np.ones(400, dtype=bool))
    assert result.p_value < 1e-10
    assert result.observed_first > result.expected_first


def test_the_log_rank_test_is_not_fooled_by_sample_size_alone():
    """Two groups from the same law, one much larger, must not separate."""
    rng = np.random.default_rng(16)
    small = rng.exponential(1.0, 120)
    large = rng.exponential(1.0, 1_200)
    result = survival.logrank(small, np.ones(120, dtype=bool),
                              large, np.ones(1_200, dtype=bool))
    assert result.p_value > 0.01


# -- competing risks ------------------------------------------------------

def test_the_incidences_and_the_survival_add_to_exactly_one():
    """The identity that catches almost every Aalen-Johansen mistake."""
    rng = np.random.default_rng(18)
    n = 2_000
    time = rng.exponential(1.0, n)
    cause = rng.choice([0, 1, 2], size=n, p=[0.3, 0.4, 0.3])

    first = survival.cumulative_incidence(time, cause, 1)
    second = survival.cumulative_incidence(time, cause, 2)
    overall = survival.kaplan_meier(time, cause > 0)

    for t in (0.1, 0.5, 1.0, 2.0, 5.0):
        total = first.at(t) + second.at(t) + overall.at(t)
        assert total == pytest.approx(1.0, abs=1e-9)


def test_ignoring_a_competing_risk_overstates_the_failure_probability():
    """One minus Kaplan-Meier is always at least the cumulative incidence.

    It has to be: removing the competing risk can only give the event
    under study more opportunity to happen.
    """
    rng = np.random.default_rng(20)
    n = 3_000
    time = rng.exponential(1.0, n)
    cause = rng.choice([0, 1, 2], size=n, p=[0.2, 0.4, 0.4])

    incidence = survival.cumulative_incidence(time, cause, 1)
    naive = survival.naive_failure_probability(time, cause == 1)
    for t in (0.25, 0.5, 1.0, 2.0):
        assert naive.at(t) >= incidence.at(t) - 1e-12
    assert naive.at(2.0) > incidence.at(2.0)


def test_with_no_competing_risk_the_two_estimators_coincide():
    rng = np.random.default_rng(22)
    time = rng.exponential(1.0, 1_000)
    cause = (rng.random(1_000) < 0.7).astype(int)   # only cause 1, or censored

    incidence = survival.cumulative_incidence(time, cause, 1)
    naive = survival.naive_failure_probability(time, cause == 1)
    for t in (0.3, 1.0, 2.0):
        assert incidence.at(t) == pytest.approx(naive.at(t), abs=1e-9)


# -- what censoring is worth ----------------------------------------------

@pytest.mark.parametrize("censoring", [0.5, 0.75, 0.9])
def test_the_censored_likelihood_recovers_a_shape_it_was_given(censoring):
    true_beta, true_eta = 1.6, 2.0
    rng = np.random.default_rng(int(censoring * 100))
    sample = true_eta * rng.weibull(true_beta, 40_000)
    limit = float(np.quantile(sample, 1.0 - censoring))
    time = np.minimum(sample, limit)
    failed = sample <= limit

    fit = weibull.fit(time, failed)
    assert fit.beta == pytest.approx(true_beta, rel=0.06)
    assert fit.eta == pytest.approx(true_eta, rel=0.08)


def test_the_naive_methods_fail_in_the_direction_the_write_up_claims():
    """Both wrong methods must be wrong, and wrong the way it says.

    Dropping the survivors discards the long lives, so the scale collapses.
    Counting them as failures makes a population that nearly all dies at
    one distance, so the shape explodes.
    """
    true_beta, true_eta = 1.6, 2.0
    rng = np.random.default_rng(90)
    sample = true_eta * rng.weibull(true_beta, 40_000)
    limit = float(np.quantile(sample, 0.10))     # 90 % censored
    time = np.minimum(sample, limit)
    failed = sample <= limit

    fits = weibull.naive_fits(time, failed)
    assert fits["censored MLE"].beta == pytest.approx(true_beta, rel=0.06)
    assert fits["drop censored"].eta < 0.5 * true_eta
    assert fits["censored as failures"].beta > 3.0 * true_beta


def test_without_censoring_all_three_methods_are_the_same_fit():
    rng = np.random.default_rng(24)
    sample = 2.0 * rng.weibull(1.6, 5_000)
    failed = np.ones_like(sample, dtype=bool)
    fits = weibull.naive_fits(sample, failed)
    betas = [fit.beta for fit in fits.values()]
    assert max(betas) - min(betas) < 1e-9


def test_a_fit_needs_at_least_two_failures():
    with pytest.raises(ValueError, match="two failures"):
        weibull.fit(np.array([1.0, 2.0, 3.0]),
                    np.array([True, False, False]))


def test_the_probability_plot_slope_recovers_the_shape():
    """An estimate of beta that owes nothing to the likelihood."""
    rng = np.random.default_rng(26)
    sample = 2.0 * rng.weibull(1.6, 4_000)
    curve = survival.kaplan_meier(sample, np.ones_like(sample, dtype=bool))
    plot = weibull.probability_plot(curve, weibull.fit(
        sample, np.ones_like(sample, dtype=bool)))
    assert plot["slope"] == pytest.approx(1.6, rel=0.12)
    assert plot["r_squared"] > 0.98


# -- turning results into lifetimes ---------------------------------------

def test_every_status_in_the_data_is_classified(entries):
    """No status may fall through to a default."""
    for entry in entries:
        assert isinstance(lifetimes.classify(entry.status), lifetimes.Cause)


def test_an_unknown_status_is_refused_rather_than_guessed():
    with pytest.raises(ValueError, match="unclassified"):
        lifetimes.classify("Sabotage by a rival team")


def test_a_crash_is_not_counted_as_a_reliability_failure(lives):
    for life in lives:
        if life.cause is lifetimes.Cause.ACCIDENT:
            assert not life.failed
        if life.cause is lifetimes.Cause.TECHNICAL:
            assert life.failed


def test_the_half_lap_convention_keeps_every_failure_off_zero(lives):
    """A car credited with no laps failed during lap one, not at time zero.

    A Weibull cannot take a failure at the origin, and dropping these would
    discard real early failures and bias the shape upward.
    """
    times = np.array([life.time for life in lives])
    assert np.all(times > 0.0)

    zero_lap_failures = [life for life in lives
                         if life.entry.laps == 0 and life.failed]
    assert len(zero_lap_failures) >= 10
    for life in zero_lap_failures:
        assert life.time == pytest.approx(0.5 / life.entry.race_laps)


def test_a_finisher_is_censored_at_its_own_distance_with_no_half_lap(lives):
    """Distance covered by a finisher is known exactly, so nothing is added."""
    for life in lives:
        if life.cause is lifetimes.Cause.FINISHED:
            assert not life.failed
            assert life.time == pytest.approx(life.entry.fraction)
    winners = [life for life in lives
               if life.entry.laps == life.entry.race_laps]
    assert winners
    assert all(life.time == pytest.approx(1.0) for life in winners)


def test_a_car_that_never_ran_is_excluded(entries, lives):
    """No exposure, so no contribution either way."""
    withdrew = [e for e in entries if e.status == "Withdrew"]
    assert withdrew
    kept = {(life.entry.season, life.entry.round, life.entry.driver)
            for life in lives}
    for entry in withdrew:
        assert (entry.season, entry.round, entry.driver) not in kept


def test_no_lifetime_runs_past_the_full_race_distance(lives):
    assert all(0.0 < life.time <= 1.0 for life in lives)


def test_the_ambiguous_assignments_are_all_documented():
    """Every judgement call has to carry its reasoning in the code."""
    for status in lifetimes.AMBIGUOUS:
        assert lifetimes.classify(status) is not None
        assert len(lifetimes.AMBIGUOUS[status]) > 20


def test_subsystem_grouping_covers_every_technical_status():
    for status in lifetimes.TECHNICAL:
        assert lifetimes.mode_of(status) in {
            "power unit", "transmission", "chassis and brakes",
            "electrics", "other"}


# -- the real data --------------------------------------------------------

def test_the_dataset_is_heavily_censored(observed):
    """The condition that makes the whole exercise necessary."""
    _, failed = observed
    assert failed.sum() > 250
    assert 1.0 - failed.mean() > 0.85


def test_the_hazard_is_indistinguishable_from_constant(observed):
    """The headline, as an interval rather than a stored number.

    beta sits near 1 and its confidence interval contains 1, so a rising
    or falling hazard over a race distance cannot be demonstrated.
    """
    time, failed = observed
    fit = weibull.fit(time, failed)
    low, high = fit.beta_confidence()
    assert 0.8 < fit.beta < 1.1
    assert low < 1.0 < high

    test = weibull.shape_test(time, failed)
    assert test["p_value"] > 0.05


def test_counting_finishers_as_failures_invents_a_wear_out_signature(
        observed):
    """The mistake does not blur the answer, it reverses it."""
    time, failed = observed
    fits = weibull.naive_fits(time, failed)
    correct, wrong = fits["censored MLE"], fits["censored as failures"]

    low, high = correct.beta_confidence()
    assert low < 1.0 < high                 # correct: cannot reject flat
    assert wrong.beta > 3.0                 # wrong: confident wear-out
    assert wrong.beta > 3.0 * correct.beta


def test_the_power_unit_is_the_one_subsystem_that_departs_from_flat(lives,
                                                                    observed):
    time, _ = observed
    mask = np.array([life.failed
                     and lifetimes.mode_of(life.status) == "power unit"
                     for life in lives])
    fit = weibull.fit(time, mask)
    _, high = fit.beta_confidence()
    assert fit.failures > 100
    assert high < 1.0           # infant mortality, interval clear of 1


def test_reliability_improved_across_the_hybrid_era(lives):
    first = [life for life in lives if life.season == 2014]
    last = [life for life in lives if life.season == 2021]

    first_curve = survival.kaplan_meier(*lifetimes.arrays(first))
    last_curve = survival.kaplan_meier(*lifetimes.arrays(last))
    assert last_curve.values[-1] > first_curve.values[-1]

    comparison = survival.logrank(*lifetimes.arrays(first),
                                  *lifetimes.arrays(last))
    assert comparison.p_value < 0.01
    assert comparison.observed_first > comparison.expected_first


def test_the_unexplained_retirements_change_nothing(entries):
    """Thirty-eight records with no cause, and two defensible treatments."""
    results = []
    for flag in (False, True):
        time, failed = lifetimes.arrays(
            lifetimes.build(entries, unknown_as_failure=flag))
        results.append(weibull.fit(time, failed))
    assert abs(results[0].beta - results[1].beta) < 0.01
    assert results[1].failures > results[0].failures


def test_the_scale_is_an_extrapolation_and_the_code_knows_it(observed):
    """eta lands far outside the observed range, which has to be flagged.

    No car runs past one race distance, so a characteristic life of eleven
    rests on the Weibull shape holding ten times beyond the data. The test
    records the fact so that a future change cannot quietly make the README
    wrong.
    """
    time, failed = observed
    fit = weibull.fit(time, failed)
    assert time.max() <= 1.0
    assert fit.eta > 5.0 * time.max()
    # The B10 life, by contrast, sits inside the data.
    assert fit.b10_life <= 1.05 * time.max()
