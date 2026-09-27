
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq


# ============================================================
# KONFIGURASI PATH DAN IMPORT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmark_utils import (
    DATASET_FILES,
    REPEAT_COUNT,
    create_result_record,
    measure_peak_rss,
    validate_dataset_files,
    write_results_csv,
)


TOOL_NAME = os.getenv("TOOL_NAME", "pandas")

OUTPUT_FILE = Path(
    os.getenv(
        "RESULTS_PATH",
        str(
            PROJECT_ROOT
            / "results"
            / "raw"
            / "pandas"
            / "benchmark_results.csv"
        ),
    )
)


# ============================================================
# PEMBACAAN DATASET
# ============================================================

def read_dataset(dataset_path: Path) -> pd.DataFrame:
    """Membaca seluruh dataset Parquet ke DataFrame Pandas."""

    return pd.read_parquet(
        dataset_path,
        engine="pyarrow",
    )


# ============================================================
# OPERASI BENCHMARK
# ============================================================

def execute_operation(
    dataset_path: Path,
    operation: str,
) -> tuple[Any, dict[str, Any]]:
    """
    Menjalankan satu operasi benchmark.

    Untuk filter, aggregation, dan groupby, waktu pembacaan
    dataset termasuk dalam pengukuran end-to-end.
    """

    if operation == "read":
        result = read_dataset(dataset_path)

        metrics = {
            "result_rows": len(result),
            "result_sum": None,
            "result_groups": None,
        }

        return result, metrics

    df = read_dataset(dataset_path)

    if operation == "filter":
        result = df.loc[
            df["trip_distance"] > 0
        ].copy()

        metrics = {
            "result_rows": len(result),
            "result_sum": None,
            "result_groups": None,
        }

        return result, metrics

    if operation == "aggregation":
        result = df["trip_distance"].sum(
            min_count=1
        )

        metrics = {
            "result_rows": None,
            "result_sum": (
                float(result)
                if pd.notna(result)
                else None
            ),
            "result_groups": None,
        }

        return result, metrics

    if operation == "groupby":
        result = (
            df.groupby(
                "PULocationID",
                dropna=False,
                sort=True,
            )["trip_distance"]
            .sum(min_count=1)
            .reset_index(name="trip_distance_sum")
        )

        valid_sums = result[
            "trip_distance_sum"
        ].dropna()

        total_sum = (
            float(valid_sums.sum(min_count=1))
            if not valid_sums.empty
            else None
        )

        metrics = {
            "result_rows": None,
            "result_sum": total_sum,
            "result_groups": len(result),
        }

        return result, metrics

    raise ValueError(
        f"Operasi tidak dikenal: {operation}"
    )


# ============================================================
# VALIDASI HASIL OPERASI
# ============================================================

def validate_operation_result(
    operation: str,
    result: Any,
    dataset_path: Path,
    metrics: dict[str, Any],
) -> None:
    """Memastikan operasi menghasilkan output yang valid."""

    if operation == "read":
        expected_rows = pq.ParquetFile(
            dataset_path
        ).metadata.num_rows

        if len(result) != expected_rows:
            raise ValueError(
                f"Read tidak valid: expected={expected_rows}, "
                f"actual={len(result)}"
            )

        if "trip_distance" not in result.columns:
            raise ValueError(
                "Kolom trip_distance tidak ditemukan."
            )

    elif operation == "filter":
        if not isinstance(result, pd.DataFrame):
            raise TypeError(
                "Hasil filter bukan DataFrame."
            )

        invalid_rows = result.loc[
            ~(result["trip_distance"] > 0)
        ]

        if not invalid_rows.empty:
            raise ValueError(
                "Hasil filter masih mengandung baris "
                "dengan trip_distance <= 0 atau null."
            )

        if len(result) != metrics["result_rows"]:
            raise ValueError(
                "Jumlah baris filter tidak konsisten."
            )

    elif operation == "aggregation":
        if metrics["result_sum"] is not None:
            if not math.isfinite(
                metrics["result_sum"]
            ):
                raise ValueError(
                    "Hasil aggregation bukan angka finite."
                )

    elif operation == "groupby":
        if not isinstance(result, pd.DataFrame):
            raise TypeError(
                "Hasil groupby bukan DataFrame."
            )

        if result["PULocationID"].duplicated().any():
            raise ValueError(
                "Hasil groupby memiliki kunci grup duplikat."
            )

        if len(result) != metrics["result_groups"]:
            raise ValueError(
                "Jumlah grup tidak konsisten."
            )


# ============================================================
# RUN BENCHMARK
# ============================================================

def run_benchmark() -> list[dict[str, Any]]:
    """Menjalankan eksperimen Pandas."""

    validate_dataset_files()

    if OUTPUT_FILE.exists():
        raise FileExistsError(
            f"File hasil sudah ada: {OUTPUT_FILE}\n"
            "Arsipkan hasil lama terlebih dahulu."
        )

    records: list[dict[str, Any]] = []

    operations = [
        "read",
        "filter",
        "aggregation",
        "groupby",
    ]

    for dataset_name, dataset_path in DATASET_FILES.items():
        dataset_size = pq.ParquetFile(
            dataset_path
        ).metadata.num_rows

        print(
            f"\n{'=' * 65}\n"
            f"DATASET: {dataset_name} "
            f"({dataset_size:,} baris)\n"
            f"{'=' * 65}"
        )

        for operation in operations:
            for repeat in range(
                1,
                REPEAT_COUNT + 1,
            ):
                print(
                    f"[{TOOL_NAME}] "
                    f"{dataset_name} | {operation} | "
                    f"repeat {repeat}/{REPEAT_COUNT}"
                )

                result, wall_time, cpu_time, memory_mb = (
                    measure_peak_rss(
                        lambda: execute_operation(
                            dataset_path,
                            operation,
                        )
                    )
                )

                output, metrics = result

                validate_operation_result(
                    operation=operation,
                    result=output,
                    dataset_path=dataset_path,
                    metrics=metrics,
                )

                records.append(
                    create_result_record(
                        tool=TOOL_NAME,
                        dataset=dataset_name,
                        dataset_size=dataset_size,
                        operation=operation,
                        repeat=repeat,
                        wall_time=wall_time,
                        cpu_time=cpu_time,
                        memory_mb=memory_mb,
                        result_rows=metrics["result_rows"],
                        result_sum=metrics["result_sum"],
                        result_groups=metrics["result_groups"],
                    )
                )

                print(
                    f"  wall={wall_time:.4f}s | "
                    f"CPU={cpu_time:.4f}s | "
                    f"sampled peak RSS delta={memory_mb:.2f} MB"
                )

    write_results_csv(
        records,
        OUTPUT_FILE,
    )

    print("\n=== BENCHMARK PANDAS SELESAI ===")
    print(f"Total eksperimen: {len(records)}")

    return records


# ============================================================
# MAIN
# ============================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Pandas pada dataset "
            "NYC Yellow Taxi."
        )
    )
    parser.parse_args()

    try:
        run_benchmark()
        return 0

    except Exception as error:
        print(
            f"\n[ERROR] {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())