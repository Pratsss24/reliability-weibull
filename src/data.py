"""Fetch Formula 1 race results and cache them.

The source is the open Jolpica mirror of the Ergast schema, which needs no
key. One endpoint carries everything this analysis needs: for each car in
each race, the laps it completed, the constructor that built it, and the
status it finished or retired under.

**Why 2014 to 2021.** Two constraints bracket the window at both ends.

The later end is a data limit, not a choice. From 2022 the feed stopped
recording *why* a car retired: 2024 has five distinct statuses and 2026 has
four, where 2021 has twenty-six and 2010 has thirty-five. A season that
reports only "Retired" cannot distinguish an engine failure from a crash,
and that distinction is the whole analysis. Checked directly against the
`/status` endpoint rather than assumed; `survey_status_granularity`
reproduces the check.

The earlier end is an engineering choice. 2014 is the first year of the
1.6-litre V6 turbo hybrid regulations, and a reliability model pooled
across a regulation change is fitting two different machines at once. The
hybrid era is internally coherent, and it is also the era in which
reliability was most obviously deciding championships.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

API = "https://api.jolpi.ca/ergast/f1/"
ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "results.json"

SEASONS = tuple(range(2014, 2022))

# The public endpoint is rate limited and answers 429 in bursts.
PAUSE = 1.3
RETRIES = 5
PAGE = 100


@dataclass(frozen=True)
class Entry:
    """One car in one race."""

    season: int
    round: int
    race: str
    circuit: str
    driver: str
    team: str
    grid: int
    laps: int                # laps this car completed
    race_laps: int           # laps the race ran to
    status: str
    position_text: str

    @property
    def fraction(self) -> float:
        """Share of the race distance this car covered, 0 to 1."""
        if self.race_laps <= 0:
            return 0.0
        return min(self.laps / self.race_laps, 1.0)


def _request(path: str, **query) -> dict:
    query.setdefault("format", "json")
    query.setdefault("limit", PAGE)
    url = API + path + "?" + "&".join(f"{k}={v}" for k, v in query.items())
    delay = PAUSE
    for attempt in range(RETRIES):
        try:
            with urllib.request.urlopen(url, timeout=40) as response:
                return json.load(response)["MRData"]
        except urllib.error.HTTPError as error:
            if error.code != 429 or attempt == RETRIES - 1:
                raise
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")


def survey_status_granularity(seasons=(2026, 2024, 2021, 2018, 2014)
                              ) -> list[dict]:
    """How many distinct retirement causes does each season record?

    The check that fixes the analysis window. A season reporting only
    "Retired" cannot support a competing-risks model, and that is a fact
    about the feed rather than about the cars.
    """
    rows = []
    for season in seasons:
        time.sleep(PAUSE)
        statuses = _request(f"{season}/status/")["StatusTable"]["Status"]
        named = [s for s in statuses
                 if s["status"] not in ("Finished", "Retired", "Lapped")
                 and not s["status"].startswith("+")]
        rows.append({
            "season": season,
            "distinct": len(statuses),
            "rows": sum(int(s["count"]) for s in statuses),
            "named_causes": len(named),
            "named_rows": sum(int(s["count"]) for s in named),
        })
    return rows


def fetch_season(season: int) -> list[Entry]:
    """Every car-race of one season.

    Results are paged, and a race can straddle a page boundary, so rows are
    accumulated per race before the race distance is taken as the greatest
    lap count any car reached.
    """
    raw: dict[tuple[int, int], dict] = {}
    offset = 0
    while True:
        time.sleep(PAUSE)
        data = _request(f"{season}/results/", limit=PAGE, offset=offset)
        races = data["RaceTable"]["Races"]
        if not races:
            break
        for race in races:
            key = (season, int(race["round"]))
            bucket = raw.setdefault(key, {"race": race, "results": []})
            bucket["results"].extend(race["Results"])
        offset += PAGE
        if offset >= int(data["total"]):
            break

    entries: list[Entry] = []
    for (_, rnd), bucket in sorted(raw.items()):
        race = bucket["race"]
        results = bucket["results"]
        race_laps = max(int(r["laps"]) for r in results)
        for result in results:
            entries.append(Entry(
                season=season, round=rnd, race=race["raceName"],
                circuit=race["Circuit"]["circuitId"],
                driver=result["Driver"]["driverId"],
                team=result["Constructor"]["constructorId"],
                grid=int(result.get("grid", 0)),
                laps=int(result["laps"]), race_laps=race_laps,
                status=result["status"],
                position_text=result["positionText"]))
    print(f"  {season}  {len(raw):2d} races  {len(entries):4d} car-races")
    return entries


def load(seasons: tuple[int, ...] = SEASONS, *,
         refresh: bool = False) -> list[Entry]:
    """Cached entries for the given seasons."""
    if CACHE.exists() and not refresh:
        raw = json.loads(CACHE.read_text(encoding="utf-8"))
        return [Entry(**record) for record in raw]

    entries: list[Entry] = []
    for season in seasons:
        entries.extend(fetch_season(season))
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps([asdict(e) for e in entries], indent=1),
                     encoding="utf-8")
    return entries


if __name__ == "__main__":
    print("status granularity by season")
    for row in survey_status_granularity():
        print(f"  {row['season']}  {row['distinct']:2d} distinct, "
              f"{row['named_causes']:2d} named causes covering "
              f"{row['named_rows']:4d} rows")
    print(f"\nfetching {SEASONS[0]}-{SEASONS[-1]}")
    collected = load(refresh=True)
    print(f"\n{len(collected)} car-races cached to {CACHE}")
