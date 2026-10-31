"""Turn race results into lifetimes: what failed, what survived, and when.

Survival analysis needs three things from every car-race, and each one is a
decision rather than a lookup.

**1. Was this a failure?** Only a technical failure of the car counts as the
event under study. A crash is not a reliability event — it is a *competing
risk*, something else that can end the car's race first and so prevent the
failure we are trying to observe. Driver illness and a pit crew error with
a wheel nut are neither. The taxonomy below is explicit about every one of
the fifty-nine statuses in the window, and `AMBIGUOUS` names the handful
whose assignment is a judgement rather than a reading.

**2. When?** The feed reports *completed* laps. A car credited with five
laps retired somewhere during lap six, so its failure time is not 5 — it is
somewhere in the interval (5, 6]. Every failure in this dataset is
interval-censored by exactly one lap, and the usual treatment is to place
it at the midpoint:

    failure time  = (laps + 0.5) / race_laps
    censoring time =  laps       / race_laps

A car that survives to the chequered flag is known to have survived its
completed laps, with no half-lap added, because that distance is certain.

The half-lap convention is not cosmetic. It is what makes the 104 cars that
completed zero laps usable — 96 of them carry exposure: they failed during
lap one, not at time zero, and a Weibull cannot take a failure at the origin
at all. Dropping them instead would discard eighteen real technical failures
and bias every estimate toward late life.

**3. On what scale?** Race fraction, not laps. Formula 1 race distance is
regulated to a minimum of 305 km, so the fraction of the winner's distance
is a reasonable proxy for distance covered and is comparable between a
44-lap Spa and a 78-lap Monaco. It is only a proxy: Monaco is the standing
exception at about 260 km, and a lap is harder on a car at some circuits
than at others. `Scope` in the README says so plainly.

A lapped finisher is censored at its own fraction, not at 1.0. It ran the
full race *time* but covered less *distance*, and on a distance scale what
is known is that it survived the distance it actually covered.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from data import Entry


class Cause(Enum):
    TECHNICAL = "technical"     # the event under study
    ACCIDENT = "accident"       # competing risk: contact, crash, debris
    DRIVER = "driver"           # competing risk: the person, not the car
    OPERATIONAL = "operational"  # pit crew or strategy, not a component
    UNKNOWN = "unknown"         # retired, cause not recorded
    FINISHED = "finished"       # right-censored at the flag
    NO_START = "no start"       # never ran: no exposure, excluded


# Every status seen in 2014-2021, assigned once and explicitly.
TECHNICAL = {
    "Engine", "Power Unit", "Gearbox", "Brakes", "Suspension", "Electrical",
    "Hydraulics", "Power loss", "Oil leak", "ERS", "Exhaust", "Turbo",
    "Battery", "Overheating", "Water pressure", "Electronics",
    "Transmission", "Fuel pressure", "Oil pressure", "Clutch", "Water leak",
    "Throttle", "Driveshaft", "Drivetrain", "Fuel system", "Brake duct",
    "Vibrations", "Spark plugs", "Steering", "Radiator", "Wheel",
    "Mechanical", "Technical",
}
ACCIDENT = {
    "Collision", "Accident", "Collision damage", "Spun off", "Damage",
    "Debris", "Puncture", "Tyre", "Front wing", "Rear wing",
}
DRIVER = {"Illness", "Seat"}
OPERATIONAL = {"Out of fuel", "Wheel nut"}
UNKNOWN = {"Retired"}
FINISHED_EXACT = {"Finished", "Disqualified", "Excluded"}

# Assignments that are a judgement call, with the reasoning, so a reader can
# disagree with a specific decision rather than with the whole taxonomy.
AMBIGUOUS = {
    "Puncture": "accident — punctures in this era come from debris or "
                "contact far more often than from a tyre defect",
    "Tyre": "accident — same reasoning; a tyre is supplied, not built by "
            "the team",
    "Front wing": "accident — a wing almost always fails after contact, "
                  "and the feed does not say whether this one did",
    "Rear wing": "accident — same reasoning as the front wing; a wing "
                 "that lets go under load has usually been hit first",
    "Wheel": "technical — a wheel or hub failure is the car's, unlike the "
             "separately coded wheel nut, which is the pit crew's",
    "Wheel nut": "operational — a pit stop error, not a component life",
    "Out of fuel": "operational — a strategy or measurement error",
    "Seat": "driver — the car was running; the person could not continue",
    "Mechanical": "technical — vague, but explicitly mechanical",
    "Retired": "unknown — kept separate, and the sensitivity of every "
               "result to its treatment is reported",
}

# A failure is interval-censored within the lap it happened on.
LAP_MIDPOINT = 0.5


@dataclass(frozen=True)
class Lifetime:
    """One car-race as a survival observation."""

    entry: Entry
    cause: Cause
    time: float              # race fraction at failure or censoring
    failed: bool             # True only for the event under study

    @property
    def season(self) -> int:
        return self.entry.season

    @property
    def team(self) -> str:
        return self.entry.team

    @property
    def status(self) -> str:
        return self.entry.status


def classify(status: str) -> Cause:
    if status in FINISHED_EXACT or status.startswith("+") or status == "Lapped":
        return Cause.FINISHED
    if status in TECHNICAL:
        return Cause.TECHNICAL
    if status in ACCIDENT:
        return Cause.ACCIDENT
    if status in DRIVER:
        return Cause.DRIVER
    if status in OPERATIONAL:
        return Cause.OPERATIONAL
    if status in UNKNOWN:
        return Cause.UNKNOWN
    if status in ("Withdrew", "Did not start", "Did not qualify",
                  "Did not prequalify", "Not classified"):
        return Cause.NO_START
    raise ValueError(f"unclassified status {status!r}")


def build(entries: list[Entry], *,
          unknown_as_failure: bool = False) -> list[Lifetime]:
    """Lifetimes for every car-race that had any exposure.

    `unknown_as_failure` decides what to do with the thirty-eight cars whose
    retirement carries no recorded cause. The default treats them as
    censored, which assumes they are no more likely to have been technical
    failures than anything else; `unknown_sensitivity` reports what the
    other choice would do.
    """
    out: list[Lifetime] = []
    for entry in entries:
        cause = classify(entry.status)
        if cause is Cause.NO_START:
            continue
        # A car that never turned a wheel has no exposure, whatever the
        # recorded reason.
        if entry.laps == 0 and cause in (Cause.DRIVER, Cause.OPERATIONAL):
            continue

        failed = cause is Cause.TECHNICAL or (
            unknown_as_failure and cause is Cause.UNKNOWN)

        if cause is Cause.FINISHED:
            time = entry.fraction
        else:
            # Retired during the lap after the last one completed.
            time = (entry.laps + LAP_MIDPOINT) / entry.race_laps
        out.append(Lifetime(entry=entry, cause=cause,
                            time=min(time, 1.0), failed=failed))
    return out


def census(lifetimes: list[Lifetime]) -> dict[str, int]:
    counts = {cause.value: 0 for cause in Cause}
    for life in lifetimes:
        counts[life.cause.value] += 1
    return {k: v for k, v in counts.items() if v}


def arrays(lifetimes: list[Lifetime]) -> tuple[np.ndarray, np.ndarray]:
    """(time, failed) as arrays, which is what every estimator wants."""
    time = np.array([life.time for life in lifetimes], dtype=float)
    failed = np.array([life.failed for life in lifetimes], dtype=bool)
    return time, failed


def mode_of(status: str) -> str:
    """Group a technical status into a subsystem.

    Coarse on purpose: with three hundred odd failures spread over thirty
    statuses, a finer split would estimate a shape parameter from four
    events and report the noise.
    """
    power = {"Engine", "Power Unit", "Turbo", "ERS", "Battery", "Exhaust",
             "Power loss", "Spark plugs", "Fuel system", "Fuel pressure",
             "Oil pressure", "Oil leak", "Overheating", "Water pressure",
             "Water leak", "Radiator"}
    transmission = {"Gearbox", "Transmission", "Clutch", "Driveshaft",
                    "Drivetrain"}
    chassis = {"Suspension", "Steering", "Wheel", "Brakes", "Brake duct",
               "Vibrations"}
    control = {"Electrical", "Electronics", "Throttle"}
    if status in power:
        return "power unit"
    if status in transmission:
        return "transmission"
    if status in chassis:
        return "chassis and brakes"
    if status in control:
        return "electrics"
    return "other"
