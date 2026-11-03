"""Weibull reliability analysis of F1 cars, 2014-2021.

    python run.py              # everything (~1 min)
    python run.py --refresh    # re-fetch the race results
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import charts        # noqa: E402
import data          # noqa: E402
import lifetimes     # noqa: E402
import report        # noqa: E402
import survival      # noqa: E402
import weibull       # noqa: E402

SUBSYSTEMS = ("power unit", "chassis and brakes", "transmission",
              "electrics", "other")
MIN_TEAM_STARTS = 100
TRUE_BETA, TRUE_ETA = 1.6, 2.0


def verdict(low: float, high: float) -> str:
    if high < 1.0:
        return "infant mortality"
    if low > 1.0:
        return "wear-out"
    return "constant hazard"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--out", default=str(ROOT / "out"))
    args = parser.parse_args()
    out_dir = Path(args.out)

    # -- the data ---------------------------------------------------------
    entries = data.load(refresh=args.refresh)
    granularity = (data.survey_status_granularity() if args.refresh
                   else [{"season": s, "distinct": d, "named_causes": c,
                          "named_rows": r}
                         for s, d, c, r in ((2026, 4, 1, 7), (2024, 5, 2, 5),
                                            (2021, 26, 21, 56),
                                            (2018, 28, 23, 83),
                                            (2014, 30, 24, 78))])

    lives = lifetimes.build(entries)
    time, failed = lifetimes.arrays(lives)
    census = lifetimes.census(lives)

    print(f"{len(entries)} car-races, {len(lives)} with exposure")
    for kind, count in census.items():
        print(f"  {kind:12s} {count:5d}  {count / len(lives):6.2%}")

    # -- the fit ----------------------------------------------------------
    fitted = weibull.fit(time, failed, name="all technical failures")
    low, high = fitted.beta_confidence()
    test = weibull.shape_test(time, failed)
    km = survival.kaplan_meier(time, failed, name="all")
    restricted = survival.restricted_mean(km)
    plot = weibull.probability_plot(km, fitted)

    print(f"\nWeibull, right-censored MLE")
    print(f"  beta {fitted.beta:.4f} +/- {fitted.beta_se:.4f}  "
          f"(95% {low:.4f}-{high:.4f})  -> {verdict(low, high)}")
    print(f"  eta  {fitted.eta:.4f} +/- {fitted.eta_se:.4f} race distances")
    print(f"  B10 life {fitted.b10_life:.4f} race distances")
    print(f"  likelihood ratio against the exponential: "
          f"chi2 {test['statistic']:.3f}, p {test['p_value']:.4f}")
    print(f"  Kaplan-Meier survival to the flag "
          f"{km.values[-1]:.4f} ({km.lower[-1]:.4f}-{km.upper[-1]:.4f})")
    print(f"  restricted mean race fraction {restricted:.4f}")
    print(f"  probability plot slope {plot['slope']:.4f} vs MLE "
          f"{plot['beta_mle']:.4f}, R2 {plot['r_squared']:.4f}")

    # -- what censoring is worth -----------------------------------------
    naive = weibull.naive_fits(time, failed, name="all")
    print("\nhandling of the censored observations")
    for name, fit in naive.items():
        print(f"  {name:22s} beta {fit.beta:7.3f}  eta {fit.eta:8.3f}  "
              f"B10 {fit.b10_life:.3f}  (n={fit.n}, d={fit.failures})")

    rng = np.random.default_rng(0)
    recovery_rows = []
    for level in (0.0, 0.25, 0.5, 0.75, 0.9, 0.95):
        sample = TRUE_ETA * rng.weibull(TRUE_BETA, 40000)
        if level > 0:
            limit = float(np.quantile(sample, 1.0 - level))
            observed = np.minimum(sample, limit)
            event = sample <= limit
        else:
            observed, event = sample, np.ones_like(sample, dtype=bool)
        fits = weibull.naive_fits(observed, event)
        row = {"censoring": level, "true_beta": TRUE_BETA,
               "true_eta": TRUE_ETA}
        for key, fit in fits.items():
            row[key] = {"beta": fit.beta, "eta": fit.eta}
        recovery_rows.append(row)

    print(f"\nrecovery of a known Weibull (beta {TRUE_BETA}, eta {TRUE_ETA})")
    for row in recovery_rows:
        print(f"  censoring {row['censoring']:4.0%}  "
              f"MLE {row['censored MLE']['beta']:6.3f}/"
              f"{row['censored MLE']['eta']:6.3f}   "
              f"dropped {row['drop censored']['beta']:7.3f}/"
              f"{row['drop censored']['eta']:6.3f}   "
              f"as-failures {row['censored as failures']['beta']:7.3f}/"
              f"{row['censored as failures']['eta']:6.3f}")

    # -- competing risks ---------------------------------------------------
    code = np.array([1 if life.cause is lifetimes.Cause.TECHNICAL
                     else (2 if life.cause is lifetimes.Cause.ACCIDENT
                           else 0) for life in lives])
    cif_technical = survival.cumulative_incidence(time, code, 1,
                                                  name="technical")
    cif_accident = survival.cumulative_incidence(time, code, 2,
                                                 name="accident")
    naive_cif = survival.naive_failure_probability(time, failed)
    all_cause = survival.kaplan_meier(time, code > 0)
    closure = (cif_technical.values[-1] + cif_accident.values[-1]
               + all_cause.values[-1])

    print("\ncompeting risks")
    print(f"  P(technical failure) Aalen-Johansen {cif_technical.values[-1]:.4f}")
    print(f"  P(technical failure) naive 1-KM     {naive_cif.values[-1]:.4f}"
          f"  (+{100 * (naive_cif.values[-1] - cif_technical.values[-1]):.2f} pts, "
          f"{naive_cif.values[-1] / cif_technical.values[-1]:.3f}x)")
    print(f"  P(accident)                         {cif_accident.values[-1]:.4f}")
    print(f"  incidences plus survival = {closure:.6f}")

    # -- subsystems --------------------------------------------------------
    subsystem_rows = []
    for mode in SUBSYSTEMS:
        mask = np.array([life.failed
                         and lifetimes.mode_of(life.status) == mode
                         for life in lives])
        if mask.sum() < 5:
            continue
        fit = weibull.fit(time, mask, name=mode)
        bounds = fit.beta_confidence()
        curve = survival.kaplan_meier(time, mask, name=mode)
        subsystem_rows.append({
            "mode": mode, "failures": int(mask.sum()), "beta": fit.beta,
            "low": bounds[0], "high": bounds[1], "eta": fit.eta,
            "survival_end": float(curve.values[-1]), "curve": curve,
            "verdict": verdict(*bounds)})

    print("\nby subsystem")
    for row in subsystem_rows:
        print(f"  {row['mode']:20s} d={row['failures']:3d}  "
              f"beta {row['beta']:.3f} ({row['low']:.3f}-{row['high']:.3f})  "
              f"S(1) {row['survival_end']:.4f}  {row['verdict']}")

    # -- seasons -----------------------------------------------------------
    season_rows = []
    for season in sorted({life.season for life in lives}):
        rows = [life for life in lives if life.season == season]
        season_time, season_failed = lifetimes.arrays(rows)
        fit = weibull.fit(season_time, season_failed, name=str(season))
        curve = survival.kaplan_meier(season_time, season_failed)
        season_rows.append({
            "season": season, "starts": len(rows),
            "failures": int(season_failed.sum()), "beta": fit.beta,
            "eta": fit.eta, "b10": fit.b10_life,
            "survival_end": float(curve.values[-1])})

    print("\nby season")
    for row in season_rows:
        print(f"  {row['season']}  {row['failures']:3d}/{row['starts']:3d}  "
              f"failure probability {1 - row['survival_end']:6.2%}  "
              f"beta {row['beta']:.3f}  B10 {row['b10']:.3f}")
    first, last = season_rows[0], season_rows[-1]
    print(f"  B10 life improved {last['b10'] / first['b10']:.2f}x from "
          f"{first['season']} to {last['season']}")

    # -- teams -------------------------------------------------------------
    counts = Counter(life.team for life in lives)
    team_rows = []
    for team, starts in counts.items():
        if starts < MIN_TEAM_STARTS:
            continue
        rows = [life for life in lives if life.team == team]
        team_time, team_failed = lifetimes.arrays(rows)
        curve = survival.kaplan_meier(team_time, team_failed, name=team)
        team_rows.append({
            "team": team, "starts": starts,
            "failures": int(team_failed.sum()),
            "survival_end": float(curve.values[-1]),
            "restricted_mean": survival.restricted_mean(curve)})

    ordered = sorted(team_rows, key=lambda r: -r["survival_end"])
    best, worst = ordered[0]["team"], ordered[-1]["team"]
    best_rows = [life for life in lives if life.team == best]
    worst_rows = [life for life in lives if life.team == worst]
    comparison = survival.logrank(*lifetimes.arrays(best_rows),
                                  *lifetimes.arrays(worst_rows),
                                  first=best, second=worst)

    print(f"\nby constructor ({len(team_rows)} with >= {MIN_TEAM_STARTS} starts)")
    for row in ordered:
        print(f"  {row['team']:14s} {row['failures']:3d}/{row['starts']:3d}  "
              f"S(1) {row['survival_end']:.4f}  "
              f"RMST {row['restricted_mean']:.4f}")
    print(f"  log-rank {best} vs {worst}: chi2 {comparison.statistic:.2f}, "
          f"p {comparison.p_value:.3e}, observed/expected "
          f"{comparison.ratio:.3f}")

    # -- the unexplained retirements --------------------------------------
    unknown_rows = []
    for flag, label in ((False, "censored (default)"), (True, "failures")):
        alt = lifetimes.build(entries, unknown_as_failure=flag)
        alt_time, alt_failed = lifetimes.arrays(alt)
        fit = weibull.fit(alt_time, alt_failed)
        curve = survival.kaplan_meier(alt_time, alt_failed)
        unknown_rows.append({
            "treatment": label, "failures": fit.failures, "beta": fit.beta,
            "eta": fit.eta, "survival_end": float(curve.values[-1])})

    print("\nthe 38 unexplained retirements")
    for row in unknown_rows:
        print(f"  treated as {row['treatment']:20s} d={row['failures']}  "
              f"beta {row['beta']:.4f}  eta {row['eta']:.3f}  "
              f"S(1) {row['survival_end']:.4f}")

    # -- output ------------------------------------------------------------
    figures = {
        "survival_fit": charts.survival_fit(km, fitted, out_dir),
        "censoring_damage": charts.censoring_damage(km, naive, out_dir),
        "recovery": charts.recovery(recovery_rows, out_dir),
        "probability_plot": charts.probability_plot(plot, out_dir),
        "competing_risks": charts.competing_risks(
            cif_technical, cif_accident, naive_cif, out_dir),
        "subsystems": charts.subsystems(subsystem_rows, out_dir),
        "seasons": charts.seasons(season_rows, out_dir),
        "teams": charts.teams(team_rows, out_dir),
    }

    report_path = report.build(
        census=census, fitted=fitted, shape_test=test, naive=naive, km=km,
        plot=plot, recovery_rows=recovery_rows,
        cif_technical=cif_technical, cif_accident=cif_accident,
        naive_cif=naive_cif, subsystem_rows=subsystem_rows,
        season_rows=season_rows, team_rows=team_rows, logrank=comparison,
        unknown_rows=unknown_rows, granularity=granularity,
        restricted=restricted, figures=figures,
        out_path=out_dir / "report.pdf")

    print(f"\nreport  {report_path}")
    for path in figures.values():
        print(f"figure  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
