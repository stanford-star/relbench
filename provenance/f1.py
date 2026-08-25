r"""Generate the **rel-f1** database from its raw source (Ergast Formula 1 data).

    python f1.py [OUT_DIR]      # default OUT_DIR: ./rel-f1

Source: a static snapshot of the Ergast F1 database, hosted by RelBench. Produces the
Hugging Face layout (manifest.yaml + db/*.parquet) reproducing stanford-star/relbench-v1/rel-f1.
"""

import sys

import numpy as np
import pandas as pd
from _lib import Table, fetch, write_hf

URL = "https://huggingface.co/datasets/stanford-star/relbench-raw/resolve/main/rel-f1/relbench-f1-raw.zip"
SHA = "2933348953b30aa9723b4831fea8071b336b74977bbcf1fb059da63a04f06eba"
VAL_TIMESTAMP, TEST_TIMESTAMP = "2005-01-01", "2010-01-01"


def build(raw) -> dict:
    circuits = pd.read_csv(raw / "circuits.csv")
    drivers = pd.read_csv(raw / "drivers.csv")
    results = pd.read_csv(raw / "results.csv")
    races = pd.read_csv(raw / "races.csv")
    standings = pd.read_csv(raw / "driver_standings.csv")
    constructors = pd.read_csv(raw / "constructors.csv")
    constructor_results = pd.read_csv(raw / "constructor_results.csv")
    constructor_standings = pd.read_csv(raw / "constructor_standings.csv")
    qualifying = pd.read_csv(raw / "qualifying.csv")

    # Drop columns that are irrelevant, leak time, or are mostly missing.
    races.drop(
        columns=[
            "url",
            "fp1_date",
            "fp1_time",
            "fp2_date",
            "fp2_time",
            "fp3_date",
            "fp3_time",
            "quali_date",
            "quali_time",
            "sprint_date",
            "sprint_time",
        ],
        inplace=True,
    )
    circuits.drop(columns=["url"], inplace=True)
    drivers.drop(
        columns=["number", "url"], inplace=True
    )  # url unique; number 803/857 null
    results.drop(
        columns=["positionText", "time", "fastestLapTime", "fastestLapSpeed"],
        inplace=True,
    )
    standings.drop(columns=["positionText"], inplace=True)
    constructors.drop(columns=["url"], inplace=True)
    constructor_standings.drop(columns=["positionText"], inplace=True)
    constructor_results.drop(columns=["status"], inplace=True)  # only 17 rows are 'D'
    qualifying.drop(columns=["q1", "q2", "q3"], inplace=True)

    # Combine race date + time into a single timestamp.
    races["time"] = races["time"].replace(r"^\\N$", "00:00:00", regex=True)
    races["date"] = pd.to_datetime(races["date"] + " " + races["time"])

    # Propagate the race timestamp to the dependent tables.
    for tbl in (
        results,
        standings,
        constructor_results,
        constructor_standings,
        qualifying,
    ):
        tbl["date"] = tbl.merge(races[["raceId", "date"]], on="raceId", how="left")[
            "date"
        ]
    # Qualifying happens the day before the race.
    qualifying["date"] = qualifying["date"] - pd.Timedelta(days=1)

    # "\N" -> NaN, and coerce the numeric result columns.
    results = results.replace(r"^\\N$", np.nan, regex=True)
    circuits = circuits.replace(r"^\\N$", np.nan, regex=True)
    circuits["alt"] = circuits["alt"].astype(float)
    for c in [
        "rank",
        "number",
        "grid",
        "position",
        "points",
        "laps",
        "milliseconds",
        "fastestLap",
    ]:
        results[c] = pd.to_numeric(results[c], errors="coerce")
    drivers["dob"] = pd.to_datetime(drivers["dob"])

    race_fk = {"raceId": "races"}
    driver_fk = {"driverId": "drivers"}
    constr_fk = {"constructorId": "constructors"}
    return {
        "races": Table(
            df=races,
            fkey_col_to_pkey_table={"circuitId": "circuits"},
            pkey_col="raceId",
            time_col="date",
        ),
        "circuits": Table(df=circuits, fkey_col_to_pkey_table={}, pkey_col="circuitId"),
        "drivers": Table(df=drivers, fkey_col_to_pkey_table={}, pkey_col="driverId"),
        "constructors": Table(
            df=constructors, fkey_col_to_pkey_table={}, pkey_col="constructorId"
        ),
        "results": Table(
            df=results,
            fkey_col_to_pkey_table={**race_fk, **driver_fk, **constr_fk},
            pkey_col="resultId",
            time_col="date",
        ),
        "standings": Table(
            df=standings,
            fkey_col_to_pkey_table={**race_fk, **driver_fk},
            pkey_col="driverStandingsId",
            time_col="date",
        ),
        "constructor_results": Table(
            df=constructor_results,
            fkey_col_to_pkey_table={**race_fk, **constr_fk},
            pkey_col="constructorResultsId",
            time_col="date",
        ),
        "constructor_standings": Table(
            df=constructor_standings,
            fkey_col_to_pkey_table={**race_fk, **constr_fk},
            pkey_col="constructorStandingsId",
            time_col="date",
        ),
        "qualifying": Table(
            df=qualifying,
            fkey_col_to_pkey_table={**race_fk, **driver_fk, **constr_fk},
            pkey_col="qualifyId",
            time_col="date",
        ),
    }


def main(out="rel-f1") -> None:
    raw = fetch(URL, SHA) / "raw"
    write_hf(out, "rel-f1", VAL_TIMESTAMP, TEST_TIMESTAMP, build(raw))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "rel-f1")
