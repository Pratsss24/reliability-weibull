"""Build the report as a printable PDF.

`paper.py` owns the page furniture; everything here is this project's own
argument, in the order the argument runs.
"""

from __future__ import annotations

from pathlib import Path

from paper import Paper


def _granularity(rows) -> tuple[list[str], list[list[str]]]:
    out = [[str(one["season"]), str(one["distinct"]),
            str(one["named_causes"]), str(one["named_rows"])]
           for one in rows]
    return (["season", "distinct statuses", "named causes",
             "rows with a named cause"], out)


def _census(census: dict) -> tuple[list[str], list[list[str]]]:
    total = sum(census.values())
    rows = [[kind, str(count), f"{count / total:.2%}"]
            for kind, count in census.items()]
    return (["classification", "car-races", "share"], rows)


def _naive(naive) -> tuple[list[str], list[list[str]]]:
    rows = [[name, f"{fit.n:,}", f"{fit.failures:,}", f"{fit.beta:.3f}",
             f"{fit.eta:.3f}", f"{fit.b10_life:.3f}"]
            for name, fit in naive.items()]
    return (["method", "observations used", "treated as failures", "beta",
             "eta", "B10 life"], rows)


def _recovery(rows) -> tuple[list[str], list[list[str]]]:
    out = [[f"{one['censoring']:.0%}",
            f"{one['censored MLE']['beta']:.3f}",
            f"{one['censored MLE']['eta']:.3f}",
            f"{one['drop censored']['beta']:.3f}",
            f"{one['drop censored']['eta']:.3f}",
            f"{one['censored as failures']['beta']:.3f}",
            f"{one['censored as failures']['eta']:.3f}"] for one in rows]
    return (["censoring", "MLE beta", "MLE eta", "dropped beta",
             "dropped eta", "as-failures beta", "as-failures eta"], out)


def _subsystems(rows) -> tuple[list[str], list[list[str]]]:
    out = [[one["mode"], str(one["failures"]), f"{one['beta']:.3f}",
            f"{one['low']:.3f} – {one['high']:.3f}",
            f"{one['survival_end']:.4f}", one["verdict"]] for one in rows]
    return (["subsystem", "failures", "beta", "95% interval",
             "survival to the flag", "verdict"], out)


def _seasons(rows) -> tuple[list[str], list[list[str]]]:
    out = [[str(one["season"]), str(one["starts"]), str(one["failures"]),
            f"{1 - one['survival_end']:.3%}", f"{one['beta']:.3f}",
            f"{one['b10']:.3f}"] for one in rows]
    return (["season", "starts", "failures", "failure probability per race",
             "beta", "B10 life"], out)


def _teams(rows) -> tuple[list[str], list[list[str]]]:
    out = [[one["team"], str(one["starts"]), str(one["failures"]),
            f"{one['survival_end']:.4f}", f"{one['restricted_mean']:.4f}"]
           for one in sorted(rows, key=lambda one: -one["survival_end"])]
    return (["constructor", "starts", "failures", "survival to the flag",
             "restricted mean"], out)


def _unknown(rows) -> tuple[list[str], list[list[str]]]:
    out = [[one["treatment"], str(one["failures"]), f"{one['beta']:.4f}",
            f"{one['eta']:.3f}", f"{one['survival_end']:.4f}"]
           for one in rows]
    return (["treatment", "failures", "beta", "eta",
             "survival to the flag"], out)


def build(*, census, fitted, shape_test, naive, km, plot, recovery_rows,
          cif_technical, cif_accident, naive_cif, subsystem_rows,
          season_rows, team_rows, logrank, unknown_rows, granularity,
          restricted, figures, out_path: Path) -> Path:
    low, high = fitted.beta_confidence()
    as_failures = naive["censored as failures"]
    dropped = naive["drop censored"]
    gap = naive_cif.values[-1] - cif_technical.values[-1]
    out = Path(out_path)
    out = out.with_name("report.pdf") if out.suffix != ".pdf" else out
    out.parent.mkdir(parents=True, exist_ok=True)

    with Paper(out, "Weibull reliability analysis of Formula 1 cars",
               f"{fitted.n:,} car-races across the 2014–2021 V6 hybrid "
               f"era, {fitted.failures} technical failures, "
               f"{fitted.censoring_rate:.1%} right-censored.") as paper:

        paper.cover(
            headline=f"Beta is {fitted.beta:.3f} — a constant hazard "
                     f"over a race distance",
            small=f"The 95% interval runs {low:.3f} to {high:.3f} and "
                  f"contains 1: an F1 car is no more likely to break in the "
                  f"last quarter of a race than in the first. A likelihood "
                  f"ratio against the exponential gives p = "
                  f"{shape_test['p_value']:.3f}, so the simpler model "
                  f"cannot be rejected.",
            tiles=[("car-races", f"{fitted.n:,}"),
                   ("technical failures", str(fitted.failures)),
                   ("censored", f"{fitted.censoring_rate:.1%}"),
                   ("shape, beta", f"{fitted.beta:.3f}")],
            blurb=f"Counting the finishers as failures instead — the "
                  f"commonest way to mishandle censored data — returns "
                  f"beta = {as_failures.beta:.2f}, a confident wear-out "
                  f"signature that is not there.",
            figure_path=figures["survival_fit"])

        paper.section(
            "Why 2014 to 2021",
            note="The later boundary is the feed, not a choice. From 2022 "
                 "the data stops recording why a car retired, and a season "
                 "that reports only \"Retired\" cannot separate an engine "
                 "failure from a crash.",
            table=_granularity(granularity),
            after="The earlier boundary is an engineering choice: 2014 "
                  "opened the V6 turbo hybrid regulations, and a "
                  "reliability model pooled across a regulation change is "
                  "fitting two different machines at once.")

        paper.section(
            "What counts as a failure",
            body="A crash is not a reliability event. It is a competing "
                 "risk: something else that can end the car's race first "
                 "and so prevent the failure under study from ever being "
                 "observed.",
            table=_census(census))

        paper.section(
            "The fitted model",
            warn=f"eta is an extrapolation. The characteristic life comes "
                 f"out at {fitted.eta:.1f} race distances, but no car in "
                 f"the data runs past one. The fit sees only the first "
                 f"{100 / fitted.eta:.0f}% of that life, so eta and "
                 f"anything derived from it — the mean life above all "
                 f"— rest on the Weibull shape holding far beyond "
                 f"where it was observed. The B10 life of "
                 f"{fitted.b10_life:.3f} sits at the edge of the data and "
                 f"is on much firmer ground.",
            table=(["quantity", "value"],
                   [["shape, beta",
                     f"{fitted.beta:.4f} ± {fitted.beta_se:.4f}"],
                    ["95% interval for beta",
                     f"{low:.4f} – {high:.4f}"],
                    ["scale, eta (race distances)",
                     f"{fitted.eta:.3f} ± {fitted.eta_se:.3f}"],
                    ["B10 life (race distances)",
                     f"{fitted.b10_life:.4f}"],
                    ["survival to the flag, Kaplan-Meier",
                     f"{km.values[-1]:.4f} ({km.lower[-1]:.4f} – "
                     f"{km.upper[-1]:.4f})"],
                    ["restricted mean race fraction", f"{restricted:.4f}"],
                    ["likelihood ratio against the exponential",
                     f"chi-squared {shape_test['statistic']:.2f}, "
                     f"p = {shape_test['p_value']:.4f}"]]))

        paper.section(
            "A check the fit does not grade itself on",
            body=f"The transformed Kaplan-Meier points give a slope of "
                 f"{plot['slope']:.3f} against the maximum likelihood "
                 f"{plot['beta_mle']:.3f}, with R-squared "
                 f"{plot['r_squared']:.4f}.",
            figure_path=figures["probability_plot"])

        paper.section(
            "What censoring is worth",
            warn=f"Neither wrong method is a rough approximation of the "
                 f"right one. Dropping the survivors throws away every long "
                 f"life and collapses the scale from {fitted.eta:.2f} to "
                 f"{dropped.eta:.2f}; counting them as failures "
                 f"manufactures a population that nearly all dies at the "
                 f"same distance, which is what beta = "
                 f"{as_failures.beta:.2f} means.",
            figure_path=figures["censoring_damage"], table=_naive(naive))

        paper.section(
            "The same three methods on data whose answer is known",
            note=f"Simulated from a Weibull with beta = "
                 f"{recovery_rows[0]['true_beta']} and eta = "
                 f"{recovery_rows[0]['true_eta']}.",
            body="At zero censoring all three agree, as they must. At the "
                 "90% censoring of the real data the censored likelihood "
                 "still recovers the truth and the other two do not.",
            figure_path=figures["recovery"], table=_recovery(recovery_rows))

        paper.section(
            "Competing risks",
            body="Treating crashes as censoring assumes a car that crashes "
                 "was just as likely to have failed later as one that did "
                 "not — and then answers the question \"what fraction "
                 "would fail if crashes were abolished\". The "
                 "Aalen-Johansen estimator answers the question actually "
                 "asked.",
            figure_path=figures["competing_risks"],
            table=(["quantity", "value"],
                   [["P(technical failure by the flag), Aalen-Johansen",
                     f"{cif_technical.values[-1]:.4f}"],
                    ["P(technical failure), naive 1 - Kaplan-Meier",
                     f"{naive_cif.values[-1]:.4f}"],
                    ["overstatement",
                     f"{100 * gap:.2f} points "
                     f"({naive_cif.values[-1] / cif_technical.values[-1]:.3f}×)"],
                    ["P(accident by the flag)",
                     f"{cif_accident.values[-1]:.4f}"]]),
            after="The correction is real but modest here, and saying so is "
                  "part of the result: the bias grows with the competing "
                  "event rate, and at a 7% crash rate there is not much of "
                  "it to grow from.")

        paper.section(
            "By subsystem",
            body="The pooled beta near 1 is a blend. The power unit is the "
                 "one subsystem whose interval clears 1 entirely, and it "
                 "does so on the infant-mortality side.",
            figure_path=figures["subsystems"],
            table=_subsystems(subsystem_rows))

        paper.section("Reliability growth across the era",
                      figure_path=figures["seasons"],
                      table=_seasons(season_rows))

        paper.section(
            "By constructor",
            note=f"Log-rank, {logrank.first} against {logrank.second}: "
                 f"chi-squared {logrank.statistic:.2f} on 1 degree of "
                 f"freedom, p = {logrank.p_value:.3e}, with "
                 f"{logrank.observed_first:.0f} failures observed against "
                 f"{logrank.expected_first:.1f} expected.",
            figure_path=figures["teams"], table=_teams(team_rows))

        paper.section(
            "The thirty-eight unexplained retirements",
            body="Neither choice moves the shape parameter past the third "
                 "decimal, so nothing in this analysis rests on it.",
            table=_unknown(unknown_rows))

    return out
