
import os
import time
import math
import gc

import dask.dataframe as dd
import psutil
import pandas as pd


DATA_DIR = os.getenv("DATA_DIR", "/workspace/data/processed")
OUTPUT_DIR = "/workspace/results"

VALUE_COL = "trip_distance"
GROUP_COL = "PULocationID"

REPEATS = 5

DATASETS = [
    ("100k", "taxi_100k.parquet", 100_000),
    ("1m", "taxi_1m.parquet", 1_000_000),
    ("full", "taxi_full.parquet", 3_530_109),
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

process = psutil.Process(os.getpid())


def measure(func):
    """Mengukur waktu, perubahan RSS, dan penggunaan CPU proses."""
    gc.collect()

    rss_before = process.memory_info().rss
    cpu_before = time.process_time()
    wall_start = time.perf_counter()

    result = func()

    wall_end = time.perf_counter()
    cpu_after = time.process_time()
    rss_after = process.memory_info().rss

    wall_seconds = wall_end - wall_start
    cpu_seconds = cpu_after - cpu_before

    memory_mb = (rss_after - rss_before) / (1024 * 1024)
    cpu_percent = (
        (cpu_seconds / wall_seconds) * 100
        if wall_seconds > 0
        else 0.0
    )

    return result, wall_seconds, memory_mb, cpu_percent


def validate_close(actual, expected):
    return math.isclose(
        float(actual),
        float(expected),
        rel_tol=1e-6,
        abs_tol=1e-6,
    )


def run_benchmark():
    rows = []

    for dataset_name, filename, expected_rows in DATASETS:
        path = os.path.join(DATA_DIR, filename)

        print(f"\n[DASK] Dataset {dataset_name}: {filename}")

        for repeat in range(1, REPEATS + 1):
            # READ
            def read_operation():
                ddf = dd.read_parquet(path)

                # Materialisasi seluruh dataset agar data benar-benar dibaca
                materialized = ddf.compute()
                row_count = len(materialized)

                del materialized

                return ddf, row_count

            (ddf, row_count), runtime, memory_mb, cpu_percent = measure(
                read_operation
            )

            read_valid = row_count == expected_rows

            rows.append({
                "tool": "dask",
                "operation": "read",
                "dataset_size": dataset_name,
                "repeat": repeat,
                "runtime_seconds": runtime,
                "memory_mb": memory_mb,
                "cpu_percent": cpu_percent,
                "output_rows": row_count,
                "validated": read_valid,
                "detail": (
                    "row_count_valid"
                    if read_valid
                    else f"expected={expected_rows}, actual={row_count}"
                ),
            })

            # FILTER
            def filter_operation():
                filtered = ddf[ddf[VALUE_COL] > 0]
                output_count = int(
                    filtered.map_partitions(len).sum().compute()
                )
                return filtered, output_count

            (filtered, filtered_count), runtime, memory_mb, cpu_percent = measure(
                filter_operation
            )

            min_filtered = filtered[VALUE_COL].min().compute()

            filter_valid = (
                filtered_count <= row_count
                and (
                    filtered_count == 0
                    or (
                        pd.notna(min_filtered)
                        and float(min_filtered) > 0
                    )
                )
            )

            rows.append({
                "tool": "dask",
                "operation": "filter",
                "dataset_size": dataset_name,
                "repeat": repeat,
                "runtime_seconds": runtime,
                "memory_mb": memory_mb,
                "cpu_percent": cpu_percent,
                "output_rows": filtered_count,
                "validated": filter_valid,
                "detail": (
                    "predicate_valid"
                    if filter_valid
                    else f"filtered_min={min_filtered}"
                ),
            })

            # AGGREGATION
            def aggregation_operation():
                return ddf[VALUE_COL].sum().compute()

            total, runtime, memory_mb, cpu_percent = measure(
                aggregation_operation
            )

            aggregation_valid = pd.notna(total) and math.isfinite(float(total))

            rows.append({
                "tool": "dask",
                "operation": "aggregation",
                "dataset_size": dataset_name,
                "repeat": repeat,
                "runtime_seconds": runtime,
                "memory_mb": memory_mb,
                "cpu_percent": cpu_percent,
                "output_rows": 1,
                "validated": bool(aggregation_valid),
                "detail": (
                    f"sum={float(total)}"
                    if aggregation_valid
                    else f"invalid_sum={total}"
                ),
            })

            # GROUP BY
            def groupby_operation():
                return ddf.groupby(GROUP_COL)[VALUE_COL].sum().compute()

            grouped, runtime, memory_mb, cpu_percent = measure(
                groupby_operation
            )

            group_sum = grouped.sum()
            group_valid = (
                pd.notna(group_sum)
                and validate_close(group_sum, total)
            )

            rows.append({
                "tool": "dask",
                "operation": "group_by",
                "dataset_size": dataset_name,
                "repeat": repeat,
                "runtime_seconds": runtime,
                "memory_mb": memory_mb,
                "cpu_percent": cpu_percent,
                "output_rows": len(grouped),
                "validated": bool(group_valid),
                "detail": (
                    "group_sum_matches_total"
                    if group_valid
                    else f"group_sum={group_sum}, total={total}"
                ),
            })

            print(
                f"  Repeat {repeat}/{REPEATS}: "
                f"rows={row_count}, "
                f"filter={filtered_count}, "
                f"group_valid={group_valid}"
            )

            del ddf, filtered, grouped
            gc.collect()

    output_path = os.path.join(OUTPUT_DIR, "benchmark_results.csv")
    pd.DataFrame(rows).to_csv(output_path, index=False)

    print(f"\ndask: {len(rows)} hasil ditulis ke {output_path}")


if __name__ == "__main__":
    run_benchmark()

