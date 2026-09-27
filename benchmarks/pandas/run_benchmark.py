import csv
import math
import os
import time
from pathlib import Path

import pandas as pd
import psutil

TOOL = "pandas"
DATA_DIR = Path(os.getenv("DATA_DIR", "/workspace/data/processed"))
RESULTS_PATH = Path(
    os.getenv("RESULTS_PATH", "/workspace/results/benchmark_results.csv")
)
REPEATS = int(os.getenv("REPEATS", "5"))

DATASETS = (
    ("100k", "taxi_100k.parquet", 100_000),
    ("1m", "taxi_1m.parquet", 1_000_000),
    ("full", "taxi_full.parquet", 3_530_109),
)

VALUE_COL = "trip_distance"
GROUP_COL = "PULocationID"

FIELDS = [
    "tool",
    "operation",
    "dataset_size",
    "repeat",
    "runtime_seconds",
    "memory_mb",
    "cpu_percent",
    "output_rows",
    "validated",
    "detail",
]


def measure(func):
    """Measure wall time, process CPU time, and RSS change."""
    process = psutil.Process()
    rss_before = process.memory_info().rss
    cpu_before = time.process_time()
    wall_before = time.perf_counter()

    result = func()

    wall_seconds = time.perf_counter() - wall_before
    cpu_seconds = time.process_time() - cpu_before
    rss_after = process.memory_info().rss

    cpu_percent = (
        cpu_seconds / wall_seconds * 100
        if wall_seconds > 0
        else 0.0
    )
    memory_mb = (rss_after - rss_before) / (1024 * 1024)

    return result, wall_seconds, memory_mb, cpu_percent


def add_result(
    rows,
    operation,
    size_name,
    repeat,
    measurement,
    output_rows,
    validated,
    detail,
):
    _, wall_seconds, memory_mb, cpu_percent = measurement

    rows.append(
        {
            "tool": TOOL,
            "operation": operation,
            "dataset_size": size_name,
            "repeat": repeat,
            "runtime_seconds": wall_seconds,
            "memory_mb": memory_mb,
            "cpu_percent": cpu_percent,
            "output_rows": output_rows,
            "validated": bool(validated),
            "detail": detail,
        }
    )


def main():
    if REPEATS < 1:
        raise ValueError("REPEATS harus minimal 1.")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for size_name, filename, expected_rows in DATASETS:
        data_path = DATA_DIR / filename

        if not data_path.is_file():
            raise FileNotFoundError(
                f"Dataset tidak ditemukan: {data_path}"
            )

        print(f"\n[PANDAS] Dataset {size_name}: {filename}")

        for repeat in range(1, REPEATS + 1):
            frame, wall, memory, cpu = measure(
                lambda: pd.read_parquet(data_path)
            )

            actual_rows = len(frame)
            read_valid = actual_rows == expected_rows

            add_result(
                rows,
                "read",
                size_name,
                repeat,
                (frame, wall, memory, cpu),
                actual_rows,
                read_valid,
                f"read_parquet({filename})",
            )

            if not read_valid:
                raise ValueError(
                    f"Jumlah baris tidak sesuai: {filename}; "
                    f"expected={expected_rows}, actual={actual_rows}"
                )

            required = {VALUE_COL, GROUP_COL}
            missing = required - set(frame.columns)
            if missing:
                raise ValueError(
                    f"Kolom wajib tidak ditemukan: {sorted(missing)}"
                )

            # FILTER: trip_distance > 0
            mask = frame[VALUE_COL] > 0

            filtered, wall, memory, cpu = measure(
                lambda: frame.loc[mask]
            )

            filter_valid = (
                len(filtered) == int(mask.sum())
                and bool((filtered[VALUE_COL] > 0).all())
            )

            add_result(
                rows,
                "filter",
                size_name,
                repeat,
                (filtered, wall, memory, cpu),
                len(filtered),
                filter_valid,
                f"{VALUE_COL} > 0",
            )

            # AGGREGATION: sum(trip_distance)
            total, wall, memory, cpu = measure(
                lambda: frame[VALUE_COL].sum()
            )

            aggregation_valid = math.isfinite(float(total))

            add_result(
                rows,
                "aggregation",
                size_name,
                repeat,
                (total, wall, memory, cpu),
                1,
                aggregation_valid,
                f"sum({VALUE_COL})",
            )

            # GROUP BY: PULocationID -> sum(trip_distance)
            grouped, wall, memory, cpu = measure(
                lambda: frame.groupby(
                    GROUP_COL,
                    dropna=False,
                )[VALUE_COL].sum()
            )

            grouped_total = grouped.sum()
            group_valid = (
                len(grouped) > 0
                and math.isclose(
                    float(grouped_total),
                    float(total),
                    rel_tol=1e-9,
                    abs_tol=1e-6,
                )
            )

            add_result(
                rows,
                "group_by",
                size_name,
                repeat,
                (grouped, wall, memory, cpu),
                len(grouped),
                group_valid,
                f"group by {GROUP_COL}, sum({VALUE_COL})",
            )

            print(
                f"  Repeat {repeat}/{REPEATS}: "
                f"rows={actual_rows}, "
                f"group_valid={group_valid}"
            )

    with RESULTS_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"\n{TOOL}: {len(rows)} hasil ditulis ke {RESULTS_PATH}"
    )


if __name__ == "__main__":
    main()
