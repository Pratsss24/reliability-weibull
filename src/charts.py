"""Figures. Committed to the repository so the write-up renders anywhere."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402
import numpy as np                     # noqa: E402

PALETTE = {"ink": "#16181d", "muted": "#5b6472", "grid": "#e3e6ea",
           "good": "#17803d", "bad": "#b4232c", "accent": "#2563a8",
           "warm": "#d9a441", "plum": "#6b4c9a"}
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.edgecolor": PALETTE["muted"], "axes.labelcolor": PALETTE["ink"],
    "text.color": PALETTE["ink"], "xtick.color": PALETTE["muted"],
    "ytick.color": PALETTE["muted"], "axes.titlesize": 10,
    "axes.titleweight": "bold", "figure.facecolor": "white",
})


def _save(fig, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _grid(ax):
    ax.grid(True, color=PALETTE["grid"], linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _step(ax, curve, colour, label, band=True, start=1.0):
    times = np.concatenate([[0.0], curve.times])
    values = np.concatenate([[start], curve.values])
    ax.step(times, values, where="post", color=colour, linewidth=1.6,
            label=label)
    if band and np.all(np.isfinite(curve.lower)):
        lower = np.concatenate([[start], curve.lower])
        upper = np.concatenate([[start], curve.upper])
        ax.fill_between(times, lower, upper, step="post", color=colour,
                        alpha=0.15, linewidth=0)


def survival_fit(km, fitted, out_dir: Path) -> Path:
    """The non-parametric curve with the fitted model over it."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    _step(axes[0], km, PALETTE["ink"], "Kaplan-Meier (95 % band)")
    grid = np.linspace(1e-4, 1.0, 400)
    axes[0].plot(grid, fitted.survival(grid), color=PALETTE["bad"],
                 linewidth=1.5, linestyle="--",
                 label=f"Weibull  beta {fitted.beta:.3f}, eta {fitted.eta:.1f}")
    axes[0].set_xlabel("race fraction completed")
    axes[0].set_ylabel("probability the car has not failed")
    axes[0].set_title("Survival against race distance", loc="left")
    axes[0].set_xlim(0, 1)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    _grid(axes[0])

    low, high = fitted.beta_confidence()
    axes[1].plot(grid, fitted.hazard(grid), color=PALETTE["accent"],
                 linewidth=1.7, label=f"beta {fitted.beta:.3f}")
    # The same scale with the shape at each end of its interval, to show
    # how little room the data leaves for a rising or falling hazard.
    for bound in (low, high):
        hazard = ((bound / fitted.eta)
                  * np.power(grid / fitted.eta, bound - 1.0))
        axes[1].plot(grid, hazard, color=PALETTE["accent"], linewidth=0.9,
                     linestyle=":")
    axes[1].axhline(1.0 / fitted.eta, color=PALETTE["muted"],
                    linestyle="--", linewidth=1.1,
                    label="constant hazard (beta = 1)")
    axes[1].set_xlabel("race fraction completed")
    axes[1].set_ylabel("hazard rate")
    axes[1].set_title(
        f"Hazard is flat: {fitted.regime}\n"
        f"95 % interval for beta, {low:.3f} to {high:.3f}", loc="left")
    axes[1].set_xlim(0, 1)
    axes[1].legend(frameon=False, fontsize=8)
    _grid(axes[1])

    fig.tight_layout()
    return _save(fig, out_dir, "survival_fit")


def censoring_damage(km, fits, out_dir: Path) -> Path:
    """What the two wrong ways of handling censoring predict."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    grid = np.linspace(1e-4, 1.0, 400)

    _step(axes[0], km, PALETTE["ink"], "Kaplan-Meier (the data)")
    colours = {"censored MLE": PALETTE["good"],
               "drop censored": PALETTE["warm"],
               "censored as failures": PALETTE["bad"]}
    for name, fit in fits.items():
        axes[0].plot(grid, fit.survival(grid), linewidth=1.5,
                     color=colours[name], linestyle="--" if name != "censored MLE" else "-",
                     label=f"{name}: beta {fit.beta:.2f}, eta {fit.eta:.2f}")
    axes[0].set_xlabel("race fraction completed")
    axes[0].set_ylabel("probability the car has not failed")
    axes[0].set_ylim(0, 1.02)
    axes[0].set_xlim(0, 1)
    axes[0].set_title("Only one of these is the data", loc="left")
    axes[0].legend(frameon=False, fontsize=7.5, loc="lower left")
    _grid(axes[0])

    names = list(fits)
    betas = [fits[n].beta for n in names]
    b10 = [fits[n].b10_life for n in names]
    x = np.arange(len(names))
    axes[1].bar(x - 0.19, betas, 0.36, color=PALETTE["accent"],
                label="shape, beta")
    axes[1].bar(x + 0.19, b10, 0.36, color=PALETTE["plum"],
                label="B10 life (race fractions)")
    axes[1].axhline(1.0, color=PALETTE["muted"], linewidth=1.0,
                    linestyle="--")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([n.replace(" ", "\n") for n in names],
                            fontsize=8)
    for index, (b, l) in enumerate(zip(betas, b10)):
        axes[1].text(index - 0.19, b + 0.06, f"{b:.2f}", ha="center",
                     fontsize=8)
        axes[1].text(index + 0.19, l + 0.06, f"{l:.2f}", ha="center",
                     fontsize=8)
    axes[1].set_title("Ignoring censoring does not add noise;\n"
                      "it reverses the engineering conclusion", loc="left")
    axes[1].legend(frameon=False, fontsize=8)
    _grid(axes[1])

    fig.tight_layout()
    return _save(fig, out_dir, "censoring_damage")


def recovery(rows, out_dir: Path) -> Path:
    """Simulation: can each method recover a shape it was given?"""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))
    levels = [r["censoring"] for r in rows]
    truth_beta = rows[0]["true_beta"]
    truth_eta = rows[0]["true_eta"]

    for key, colour, marker in (("censored MLE", PALETTE["good"], "o"),
                                ("drop censored", PALETTE["warm"], "s"),
                                ("censored as failures", PALETTE["bad"], "^")):
        axes[0].plot(levels, [r[key]["beta"] for r in rows], marker=marker,
                     markersize=4, color=colour, linewidth=1.3, label=key)
        axes[1].plot(levels, [r[key]["eta"] for r in rows], marker=marker,
                     markersize=4, color=colour, linewidth=1.3, label=key)

    axes[0].axhline(truth_beta, color=PALETTE["ink"], linestyle="--",
                    linewidth=1.1, label=f"truth, beta = {truth_beta}")
    axes[1].axhline(truth_eta, color=PALETTE["ink"], linestyle="--",
                    linewidth=1.1, label=f"truth, eta = {truth_eta}")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("estimated shape, beta")
    axes[1].set_ylabel("estimated scale, eta")
    for ax, title in zip(axes, ("Shape", "Scale")):
        ax.set_xlabel("fraction of observations censored")
        ax.set_title(f"{title}: recovery of a known Weibull", loc="left")
        ax.legend(frameon=False, fontsize=7.5)
        _grid(ax)

    fig.tight_layout()
    return _save(fig, out_dir, "recovery")


def probability_plot(plot, out_dir: Path) -> Path:
    """ln(-ln S) against ln t: a straight line if the Weibull holds."""
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.scatter(plot["x"], plot["y"], s=12, color=PALETTE["ink"],
               alpha=0.65, label="Kaplan-Meier points")
    line = np.linspace(plot["x"].min(), plot["x"].max(), 50)
    ax.plot(line, plot["slope"] * line + plot["intercept"],
            color=PALETTE["bad"], linewidth=1.5,
            label=f"least squares, slope {plot['slope']:.3f}")
    ax.plot(line, plot["beta_mle"] * line
            + (plot["intercept"] + (plot["slope"] - plot["beta_mle"])
               * line.mean()),
            color=PALETTE["accent"], linewidth=1.3, linestyle="--",
            label=f"MLE slope {plot['beta_mle']:.3f}")
    ax.set_xlabel("ln(race fraction)")
    ax.set_ylabel("ln(-ln S)")
    ax.set_title(f"Weibull probability plot, R-squared {plot['r_squared']:.4f}\n"
                 "straight means the Weibull assumption holds", loc="left")
    ax.legend(frameon=False, fontsize=8)
    _grid(ax)
    return _save(fig, out_dir, "probability_plot")


def competing_risks(cif_technical, cif_accident, naive, out_dir: Path) -> Path:
    """Cumulative incidence against the estimator that ignores the risk."""
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    _step(ax, naive, PALETTE["bad"], "1 - Kaplan-Meier (treats crashes "
          "as censoring)", band=False, start=0.0)
    _step(ax, cif_technical, PALETTE["accent"],
          "technical failure, Aalen-Johansen", band=False, start=0.0)
    _step(ax, cif_accident, PALETTE["warm"], "accident, Aalen-Johansen",
          band=False, start=0.0)
    ax.set_xlabel("race fraction completed")
    ax.set_ylabel("probability the car has met this fate")
    ax.set_xlim(0, 1)
    gap = naive.values[-1] - cif_technical.values[-1]
    ax.set_title("Competing risks: a crash stops the clock on a failure\n"
                 f"ignoring that overstates failure probability by "
                 f"{100 * gap:.2f} points", loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    _grid(ax)
    return _save(fig, out_dir, "competing_risks")


def subsystems(rows, out_dir: Path) -> Path:
    """Shape parameter by subsystem, with intervals."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))

    names = [r["mode"] for r in rows]
    betas = [r["beta"] for r in rows]
    low = [r["beta"] - r["low"] for r in rows]
    high = [r["high"] - r["beta"] for r in rows]
    y = np.arange(len(names))

    colours = [PALETTE["bad"] if r["high"] < 1.0
               else (PALETTE["good"] if r["low"] > 1.0 else PALETTE["muted"])
               for r in rows]
    for index, colour in enumerate(colours):
        axes[0].errorbar(betas[index], y[index],
                         xerr=[[low[index]], [high[index]]], fmt="o",
                         color=colour, markersize=6, capsize=3,
                         linewidth=1.3)
    axes[0].axvline(1.0, color=PALETTE["ink"], linewidth=1.1,
                    linestyle="--")
    axes[0].text(1.02, -0.45, "constant hazard", fontsize=7.5,
                 color=PALETTE["muted"])
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([f"{r['mode']}\n{r['failures']} failures"
                             for r in rows], fontsize=8)
    axes[0].set_xlabel("shape parameter, beta  (95 % interval)")
    axes[0].set_title("Only the power unit departs from constant hazard",
                      loc="left")
    _grid(axes[0])

    for row, colour in zip(rows, (PALETTE["bad"], PALETTE["accent"],
                                  PALETTE["good"], PALETTE["warm"],
                                  PALETTE["plum"])):
        curve = row["curve"]
        _step(axes[1], curve, colour, f"{row['mode']} ({row['failures']})",
              band=False)
    axes[1].set_xlabel("race fraction completed")
    axes[1].set_ylabel("survival")
    axes[1].set_xlim(0, 1)
    axes[1].set_title("Survival by subsystem", loc="left")
    axes[1].legend(frameon=False, fontsize=7.5, loc="lower left")
    _grid(axes[1])

    fig.tight_layout()
    return _save(fig, out_dir, "subsystems")


def seasons(rows, out_dir: Path) -> Path:
    """Reliability growth across the hybrid era."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    years = [r["season"] for r in rows]

    axes[0].plot(years, [100 * (1 - r["survival_end"]) for r in rows],
                 marker="o", markersize=5, color=PALETTE["bad"],
                 linewidth=1.6)
    axes[0].set_ylabel("technical failures per 100 starts")
    axes[0].set_xlabel("season")
    axes[0].set_title("Failure probability over one race distance",
                      loc="left")
    _grid(axes[0])

    axes[1].plot(years, [r["b10"] for r in rows], marker="s", markersize=5,
                 color=PALETTE["accent"], linewidth=1.6)
    axes[1].axhline(1.0, color=PALETTE["muted"], linestyle="--",
                    linewidth=1.0)
    axes[1].text(years[0], 1.04, "one race distance", fontsize=7.5,
                 color=PALETTE["muted"])
    axes[1].set_ylabel("B10 life (race distances)")
    axes[1].set_xlabel("season")
    axes[1].set_title("Distance by which a tenth of the field has failed",
                      loc="left")
    _grid(axes[1])

    fig.tight_layout()
    return _save(fig, out_dir, "seasons")


def teams(rows, out_dir: Path) -> Path:
    """Survival to the flag, by constructor."""
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ordered = sorted(rows, key=lambda r: r["survival_end"])
    names = [r["team"] for r in ordered]
    values = [100 * r["survival_end"] for r in ordered]
    y = np.arange(len(names))

    ax.barh(y, values, color=PALETTE["accent"], height=0.66)
    for index, row in enumerate(ordered):
        ax.text(values[index] + 0.3, index,
                f"{row['failures']}/{row['starts']}", va="center",
                fontsize=7.5, color=PALETTE["muted"])
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlim(80, 100)
    ax.set_xlabel("probability of reaching the flag without a technical "
                  "failure (%)")
    ax.set_title("Constructors, 2014-2021\n"
                 "labels are failures over starts", loc="left")
    _grid(ax)
    return _save(fig, out_dir, "teams")
