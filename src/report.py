"""One self-contained HTML file: charts inlined, no assets, no server."""

from __future__ import annotations

import base64
from pathlib import Path

CSS = """
body{font:15px/1.6 -apple-system,Segoe UI,Roboto,sans-serif;max-width:1120px;
margin:40px auto;padding:0 20px;color:#16181d}
h1{margin-bottom:4px}h2{margin-top:38px;border-bottom:1px solid #e3e6ea;
padding-bottom:6px}
.sub{color:#5b6472;margin-top:0}
.tiles{display:flex;gap:14px;flex-wrap:wrap;margin:22px 0}
.tile{flex:1;min-width:150px;background:#f6f8fa;border:1px solid #e3e6ea;
border-radius:9px;padding:14px 16px}
.tile .v{font-size:25px;font-weight:650;letter-spacing:-.5px}
.tile .k{font-size:11px;text-transform:uppercase;letter-spacing:.07em;
color:#5b6472;margin-top:3px}
table{border-collapse:collapse;width:100%;margin:14px 0;font-size:14px}
th,td{border:1px solid #e3e6ea;padding:7px 10px;text-align:right}
th:first-child,td:first-child{text-align:left}
th{background:#f6f8fa;font-weight:600}
td.good{color:#17803d}td.bad{color:#b4232c}
img{width:100%;margin:14px 0;border:1px solid #e3e6ea;border-radius:7px}
.note{background:#fff8e6;border-left:3px solid #d9a441;padding:11px 15px;
margin:18px 0;font-size:14px}
code{background:#f0f2f5;padding:1px 5px;border-radius:4px;font-size:13px}
"""


def _img(path: Path) -> str:
    data = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f'<img src="data:image/png;base64,{data}" alt="{Path(path).stem}">'


def build(*, census, fitted, shape_test, naive, km, plot, recovery_rows,
          cif_technical, cif_accident, naive_cif, subsystem_rows,
          season_rows, team_rows, logrank, unknown_rows, granularity,
          restricted, figures, out_path: Path) -> Path:
    low, high = fitted.beta_confidence()
    as_failures = naive["censored as failures"]
    dropped = naive["drop censored"]

    tiles = [
        ("car-races", f"{fitted.n:,}"),
        ("technical failures", str(fitted.failures)),
        ("censored", f"{fitted.censoring_rate:.1%}"),
        ("shape, beta", f"{fitted.beta:.3f}"),
    ]
    tiles_html = "".join(
        f'<div class="tile"><div class="v">{v}</div><div class="k">{k}</div></div>'
        for k, v in tiles)

    granularity_rows = "".join(
        f"<tr><td>{r['season']}</td><td>{r['distinct']}</td>"
        f'<td class="{"bad" if r["named_causes"] < 5 else "good"}">'
        f"{r['named_causes']}</td><td>{r['named_rows']}</td></tr>"
        for r in granularity)

    census_rows = "".join(
        f"<tr><td>{kind}</td><td>{count}</td>"
        f"<td>{count / sum(census.values()):.2%}</td></tr>"
        for kind, count in census.items())

    naive_rows = "".join(
        f"<tr><td>{name}</td><td>{fit.n:,}</td><td>{fit.failures:,}</td>"
        f'<td class="{"good" if name == "censored MLE" else "bad"}">'
        f"{fit.beta:.3f}</td><td>{fit.eta:.3f}</td>"
        f"<td>{fit.b10_life:.3f}</td></tr>"
        for name, fit in naive.items())

    recovery_rows_html = "".join(
        f"<tr><td>{r['censoring']:.0%}</td>"
        f'<td class="good">{r["censored MLE"]["beta"]:.3f}</td>'
        f'<td class="good">{r["censored MLE"]["eta"]:.3f}</td>'
        f'<td class="bad">{r["drop censored"]["beta"]:.3f}</td>'
        f'<td class="bad">{r["drop censored"]["eta"]:.3f}</td>'
        f'<td class="bad">{r["censored as failures"]["beta"]:.3f}</td>'
        f'<td class="bad">{r["censored as failures"]["eta"]:.3f}</td></tr>'
        for r in recovery_rows)

    subsystem_html = "".join(
        f"<tr><td>{r['mode']}</td><td>{r['failures']}</td>"
        f'<td class="{"bad" if r["high"] < 1 else ("good" if r["low"] > 1 else "")}">'
        f"{r['beta']:.3f}</td><td>{r['low']:.3f} &ndash; {r['high']:.3f}</td>"
        f"<td>{r['survival_end']:.4f}</td><td>{r['verdict']}</td></tr>"
        for r in subsystem_rows)

    season_html = "".join(
        f"<tr><td>{r['season']}</td><td>{r['starts']}</td>"
        f"<td>{r['failures']}</td><td>{1 - r['survival_end']:.3%}</td>"
        f"<td>{r['beta']:.3f}</td><td>{r['b10']:.3f}</td></tr>"
        for r in season_rows)

    team_html = "".join(
        f"<tr><td>{r['team']}</td><td>{r['starts']}</td>"
        f"<td>{r['failures']}</td><td>{r['survival_end']:.4f}</td>"
        f"<td>{r['restricted_mean']:.4f}</td></tr>"
        for r in sorted(team_rows, key=lambda r: -r["survival_end"]))

    unknown_html = "".join(
        f"<tr><td>{r['treatment']}</td><td>{r['failures']}</td>"
        f"<td>{r['beta']:.4f}</td><td>{r['eta']:.3f}</td>"
        f"<td>{r['survival_end']:.4f}</td></tr>"
        for r in unknown_rows)

    gap = naive_cif.values[-1] - cif_technical.values[-1]

    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Reliability Weibull</title><style>{CSS}</style></head><body>
<h1>Weibull reliability analysis of Formula 1 cars</h1>
<p class="sub">{fitted.n:,} car-races across the 2014&ndash;2021 V6 hybrid
era, {fitted.failures} technical failures, {fitted.censoring_rate:.1%}
right-censored.</p>
<div class="tiles">{tiles_html}</div>

<div class="note"><b>The headline.</b> The fitted shape parameter is
<b>beta = {fitted.beta:.3f}</b>, with a 95 % interval of {low:.3f} to
{high:.3f} that contains 1. Over a race distance an F1 car has an
essentially <b>constant hazard</b>: it is no more likely to break in the
last quarter of a race than in the first. A likelihood ratio against the
exponential gives p = {shape_test['p_value']:.3f}, so the simpler model
cannot be rejected. Counting the finishers as failures instead &mdash; the
commonest way to mishandle censored data &mdash; returns
beta = {as_failures.beta:.2f}, a confident wear-out signature that is not
there.</div>

<h2>Why 2014 to 2021</h2>
<p>The later boundary is the feed, not a choice. From 2022 the data stops
recording why a car retired, and a season that reports only "Retired"
cannot separate an engine failure from a crash.</p>
<table><tr><th>season</th><th>distinct statuses</th>
<th>named causes</th><th>rows with a named cause</th></tr>
{granularity_rows}</table>
<p>The earlier boundary is an engineering choice: 2014 opened the V6 turbo
hybrid regulations, and a reliability model pooled across a regulation
change is fitting two different machines at once.</p>

<h2>What counts as a failure</h2>
<table><tr><th>classification</th><th>car-races</th><th>share</th></tr>
{census_rows}</table>
<p>A crash is not a reliability event. It is a competing risk: something
else that can end the car's race first and so prevent the failure under
study from ever being observed.</p>

<h2>Survival, and the fitted model</h2>
{_img(figures['survival_fit'])}
<table><tr><th>quantity</th><th>value</th></tr>
<tr><td>shape, beta</td><td>{fitted.beta:.4f} &plusmn; {fitted.beta_se:.4f}</td></tr>
<tr><td>95 % interval for beta</td><td>{low:.4f} &ndash; {high:.4f}</td></tr>
<tr><td>scale, eta (race distances)</td>
<td>{fitted.eta:.3f} &plusmn; {fitted.eta_se:.3f}</td></tr>
<tr><td>B10 life (race distances)</td><td>{fitted.b10_life:.4f}</td></tr>
<tr><td>survival to the flag, Kaplan-Meier</td>
<td>{km.values[-1]:.4f} ({km.lower[-1]:.4f} &ndash; {km.upper[-1]:.4f})</td></tr>
<tr><td>restricted mean race fraction</td><td>{restricted:.4f}</td></tr>
<tr><td>likelihood ratio against the exponential</td>
<td>chi-squared {shape_test['statistic']:.2f}, p = {shape_test['p_value']:.4f}</td></tr>
</table>
<div class="note"><b>eta is an extrapolation.</b> The characteristic life
comes out at {fitted.eta:.1f} race distances, but no car in the data runs
past one. The fit sees only the first
{100 / fitted.eta:.0f} % of that life, so eta and anything derived from it
&mdash; the mean life above all &mdash; rest on the Weibull shape holding
far beyond where it was observed. The B10 life of
{fitted.b10_life:.3f} sits at the edge of the data and is on much firmer
ground.</div>
{_img(figures['probability_plot'])}
<p>The probability plot is a check the fit does not grade itself on: the
transformed Kaplan-Meier points give a slope of {plot['slope']:.3f} against
the maximum likelihood {plot['beta_mle']:.3f}, with
R-squared {plot['r_squared']:.4f}.</p>

<h2>What censoring is worth</h2>
<table><tr><th>method</th><th>observations used</th><th>treated as failures</th>
<th>beta</th><th>eta</th><th>B10 life</th></tr>{naive_rows}</table>
{_img(figures['censoring_damage'])}
<p>Neither wrong method is a rough approximation of the right one.
Dropping the survivors throws away every long life and collapses the scale
from {fitted.eta:.2f} to {dropped.eta:.2f}; counting them as failures
manufactures a population that nearly all dies at the same distance, which
is what beta = {as_failures.beta:.2f} means.</p>

<h3>The same three methods on data whose answer is known</h3>
<table><tr><th>censoring</th><th>MLE beta</th><th>MLE eta</th>
<th>dropped beta</th><th>dropped eta</th><th>as-failures beta</th>
<th>as-failures eta</th></tr>{recovery_rows_html}</table>
{_img(figures['recovery'])}
<p>Simulated from a Weibull with beta =
{recovery_rows[0]['true_beta']} and eta = {recovery_rows[0]['true_eta']}.
At zero censoring all three agree, as they must. At the 90 % censoring of
the real data the censored likelihood still recovers the truth and the
other two do not.</p>

<h2>Competing risks</h2>
{_img(figures['competing_risks'])}
<p>Treating crashes as censoring assumes a car that crashes was just as
likely to have failed later as one that did not &mdash; and then answers
the question "what fraction would fail if crashes were abolished". The
Aalen-Johansen estimator answers the question actually asked.</p>
<table><tr><th>quantity</th><th>value</th></tr>
<tr><td>P(technical failure by the flag), Aalen-Johansen</td>
<td>{cif_technical.values[-1]:.4f}</td></tr>
<tr><td>P(technical failure), naive 1 &minus; Kaplan-Meier</td>
<td class="bad">{naive_cif.values[-1]:.4f}</td></tr>
<tr><td>overstatement</td>
<td>{100 * gap:.2f} points ({naive_cif.values[-1] / cif_technical.values[-1]:.3f}&times;)</td></tr>
<tr><td>P(accident by the flag)</td><td>{cif_accident.values[-1]:.4f}</td></tr>
</table>
<p>The correction is real but modest here, and saying so is part of the
result: the bias grows with the competing event rate, and at a 7 % crash
rate there is not much of it to grow from.</p>

<h2>By subsystem</h2>
{_img(figures['subsystems'])}
<table><tr><th>subsystem</th><th>failures</th><th>beta</th>
<th>95 % interval</th><th>survival to the flag</th><th>verdict</th></tr>
{subsystem_html}</table>
<p>The pooled beta near 1 is a blend. The power unit is the one subsystem
whose interval clears 1 entirely, and it does so on the infant-mortality
side.</p>

<h2>Reliability growth across the era</h2>
{_img(figures['seasons'])}
<table><tr><th>season</th><th>starts</th><th>failures</th>
<th>failure probability per race</th><th>beta</th><th>B10 life</th></tr>
{season_html}</table>

<h2>By constructor</h2>
{_img(figures['teams'])}
<table><tr><th>constructor</th><th>starts</th><th>failures</th>
<th>survival to the flag</th><th>restricted mean</th></tr>{team_html}</table>
<p>Log-rank, {logrank.first} against {logrank.second}: chi-squared
{logrank.statistic:.2f} on 1 degree of freedom, p =
{logrank.p_value:.3e}, with {logrank.observed_first:.0f} failures observed
against {logrank.expected_first:.1f} expected.</p>

<h2>The thirty-eight unexplained retirements</h2>
<table><tr><th>treatment</th><th>failures</th><th>beta</th><th>eta</th>
<th>survival to the flag</th></tr>{unknown_html}</table>
<p>Neither choice moves the shape parameter past the third decimal, so
nothing in this analysis rests on it.</p>
</body></html>"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path
