
import os
import sys
from pathlib import Path

import dask.dataframe as dd
import numpy as np

# ============================================================
# SETUP PATH DAN IMPORT UTILITAS
# ============================================================

PROJECT_ROOT = Path("/workspace")

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from benchmark_utils import (
    DATASET_FILES,
    REPEAT_COUNT,
    create_result_record,
    measure_peak_rss,
    write_results_csv,
)

TOOL_NAME = os.getenv("TOOL_NAME", "dask")

RESULTS_PATH = Path(
    os.getenv(
        "RESULTS_PATH",
        "/workspace/results/raw/dask/benchmark_results.csv",
    )
)

SCHEDULER = "threads"


# ============================================================
# FUNGSI OPERASI BENCHMARK
# ============================================================

def read_dataset(file_path):
    """Membaca seluruh dataset Parquet."""
    df = dd.read_parquet(file_path)
    return df.compute(scheduler=SCHEDULER)


def filter_dataset(file_path):
    """Memfilter baris dengan trip_distance > 0."""
    df = dd.read_parquet(file_path)

    return df[df["trip_distance"] > 0].compute(
        scheduler=SCHEDULER
    )


def aggregate_dataset(file_path):
    """Menghitung total trip_distance."""
    df = dd.read_parquet(file_path)

    result = df["trip_distance"].sum().compute(
        scheduler=SCHEDULER
    )

    return float(result)


def groupby_dataset(file_path):
    """Mengelompokkan trip_distance berdasarkan PULocationID."""
    df = dd.read_parquet(file_path)

    result = (
        df.groupby("PULocationID")["trip_distance"]
        .sum()
        .compute(scheduler=SCHEDULER)
        .reset_index()
    )

    result.columns = ["PULocationID", "trip_distance_sum"]

    return result


# ============================================================
# VALIDASI HASIL
# ============================================================

def validate_operation(operation, result, dataset_size):
    """Memvalidasi hasil berdasarkan jenis operasi."""

    if operation == "read":
        valid = len(result) == dataset_size
        return "success" if valid else "failed"

    elif operation == "filter":
        valid = (
            len(result) <= dataset_size
            and (result["trip_distance"] > 0).all()
        )
        return "success" if valid else "failed"

    elif operation == "aggregation":
        valid = np.isfinite(result)
        return "success" if valid else "failed"

    elif operation == "groupby":
        valid = (
            result["PULocationID"].is_unique
            and result["trip_distance_sum"].notna().all()
        )
        return "success" if valid else "failed"

    return "failed"


# ============================================================
# EKSEKUSI BENCHMARK
# ============================================================

def run_benchmark():
    records = []

    print("=" * 70)
    print("BENCHMARK DASK")
    print("=" * 70)

    print(f"Jumlah dataset : {len(DATASET_FILES)}")
    print(f"Jumlah repeat  : {REPEAT_COUNT}")
    print(f"Scheduler      : {SCHEDULER}")

    operations = {
        "read": read_dataset,
        "filter": filter_dataset,
        "aggregation": aggregate_dataset,
        "groupby": groupby_dataset,
    }

    for dataset_name, file_path in DATASET_FILES.items():
        print(f"\nDataset: {dataset_name}")
        print(f"File   : {file_path}")

        if not Path(file_path).exists():
            raise FileNotFoundError(
                f"Dataset tidak ditemukan: {file_path}"
            )

        # Menghitung jumlah baris dataset.
        dataset = dd.read_parquet(file_path)
        dataset_size = len(dataset)

        print(f"Jumlah baris: {dataset_size:,}")

        for operation_name, operation_func in operations.items():
            print(f"\nOperasi: {operation_name}")

            for repeat in range(1, REPEAT_COUNT + 1):

                result, wall_time, cpu_time, memory_mb = (
                    measure_peak_rss(
                        lambda: operation_func(file_path)
                    )
                )

                # --------------------------------------------
                # METADATA HASIL
                # --------------------------------------------

                if operation_name == "read":
                    result_rows = len(result)
                    result_sum = None
                    result_groups = None

                elif operation_name == "filter":
                    result_rows = len(result)
                    result_sum = None
                    result_groups = None

                elif operation_name == "aggregation":
                    result_rows = 1
                    result_sum = float(result)
                    result_groups = None

                elif operation_name == "groupby":
                    result_rows = len(result)
                    result_sum = float(
                        result["trip_distance_sum"].sum()
                    )
                    result_groups = len(result)

                # --------------------------------------------
                # VALIDASI HASIL
                # --------------------------------------------

                validation_status = validate_operation(
                    operation=operation_name,
                    result=result,
                    dataset_size=dataset_size,
                )

                # --------------------------------------------
                # MEMBUAT RECORD BENCHMARK
                # --------------------------------------------

                record = create_result_record(
                    tool=TOOL_NAME,
                    dataset=dataset_name,
                    dataset_size=dataset_size,
                    operation=operation_name,
                    repeat=repeat,
                    wall_time=wall_time,
                    cpu_time=cpu_time,
                    memory_mb=memory_mb,
                    result_rows=result_rows,
                    result_sum=result_sum,
                    result_groups=result_groups,
                    validation_status=validation_status,
                )

                records.append(record)

                print(
                    f"  Repeat {repeat}/{REPEAT_COUNT} | "
                    f"Wall: {wall_time:.4f}s | "
                    f"CPU: {cpu_time:.4f}s | "
                    f"RSS delta: {memory_mb:.2f} MB | "
                    f"Status: {validation_status}"
                )

    # ========================================================
    # SIMPAN HASIL
    # ========================================================

    write_results_csv(records, RESULTS_PATH)

    print("\n" + "=" * 70)
    print("BENCHMARK DASK SELESAI")
    print("=" * 70)
    print(f"Hasil benchmark disimpan: {RESULTS_PATH}")
    print(f"Jumlah record: {len(records)}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    run_benchmark()