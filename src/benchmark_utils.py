
from __future__ import annotations

import csv
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable

import psutil


# ============================================================
# KONFIGURASI BENCHMARK
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_FILES = {
    "100k": PROJECT_ROOT / "data" / "processed" / "taxi_100k.parquet",
    "1m": PROJECT_ROOT / "data" / "processed" / "taxi_1m.parquet",
    "full": PROJECT_ROOT / "data" / "processed" / "taxi_full.parquet",
}

REPEAT_COUNT = int(os.getenv("REPEATS", "5"))

CSV_COLUMNS = [
    "tool",
    "dataset",
    "dataset_size",
    "operation",
    "repeat",
    "wall_time_seconds",
    "cpu_time_seconds",
    "memory_mb",
    "memory_metric",
    "result_rows",
    "result_sum",
    "result_groups",
    "validation_status",
]


# ============================================================
# PENGUKURAN PEAK RSS
# ============================================================

def measure_peak_rss(
    operation: Callable[[], Any],
    interval_seconds: float = 0.005,
) -> tuple[Any, float, float, float]:
    """
    Menjalankan operasi dan mengukur:
    - wall-clock time
    - CPU time proses
    - kenaikan peak RSS proses selama operasi

    memory_mb adalah peak RSS dikurangi RSS awal.
    Sampling RSS dilakukan secara berkala.

    Catatan:
    Pengukuran dilakukan dalam proses Python yang sama.
    Hasil dapat dipengaruhi oleh alokasi memori sebelumnya.
    """

    if interval_seconds <= 0:
        raise ValueError(
            "interval_seconds harus lebih besar dari 0."
        )

    process = psutil.Process(os.getpid())

    baseline_rss = process.memory_info().rss
    peak_rss = [baseline_rss]
    stop_event = threading.Event()

    def sample_memory() -> None:
        while not stop_event.is_set():
            try:
                current_rss = process.memory_info().rss
                peak_rss[0] = max(
                    peak_rss[0],
                    current_rss,
                )
            except psutil.Error:
                pass

            stop_event.wait(interval_seconds)

    sampler = threading.Thread(
        target=sample_memory,
        daemon=True,
    )

    sampler.start()

    wall_start = time.perf_counter()
    cpu_start = time.process_time()

    try:
        result = operation()
    finally:
        wall_end = time.perf_counter()
        cpu_end = time.process_time()

        stop_event.set()
        sampler.join()

        try:
            peak_rss[0] = max(
                peak_rss[0],
                process.memory_info().rss,
            )
        except psutil.Error:
            pass

    wall_time = wall_end - wall_start
    cpu_time = cpu_end - cpu_start

    peak_memory_mb = max(
        0.0,
        (peak_rss[0] - baseline_rss) / (1024 * 1024),
    )

    return (
        result,
        wall_time,
        cpu_time,
        peak_memory_mb,
    )


# ============================================================
# VALIDASI DATASET
# ============================================================

def validate_dataset_files() -> None:
    """Memastikan seluruh dataset preprocessing tersedia."""

    missing_files = [
        str(path)
        for path in DATASET_FILES.values()
        if not path.is_file()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Dataset berikut belum tersedia:\n"
            + "\n".join(missing_files)
        )


# ============================================================
# PEMBENTUKAN RECORD HASIL
# ============================================================

def create_result_record(
    *,
    tool: str,
    dataset: str,
    dataset_size: int,
    operation: str,
    repeat: int,
    wall_time: float,
    cpu_time: float,
    memory_mb: float,
    result_rows: int | None = None,
    result_sum: float | None = None,
    result_groups: int | None = None,
    validation_status: str = "passed",
) -> dict[str, Any]:
    """Membuat satu record dengan skema CSV yang seragam."""

    return {
        "tool": tool,
        "dataset": dataset,
        "dataset_size": dataset_size,
        "operation": operation,
        "repeat": repeat,
        "wall_time_seconds": wall_time,
        "cpu_time_seconds": cpu_time,
        "memory_mb": memory_mb,
        "memory_metric": "sampled_peak_rss_delta",
        "result_rows": result_rows,
        "result_sum": result_sum,
        "result_groups": result_groups,
        "validation_status": validation_status,
    }


# ============================================================
# PENYIMPANAN CSV
# ============================================================

def write_results_csv(
    records: list[dict[str, Any]],
    output_file: Path,
) -> None:
    """
    Menyimpan hasil benchmark ke CSV.

    File yang sudah ada tidak ditimpa secara diam-diam.
    """

    if not records:
        raise ValueError(
            "Tidak ada hasil benchmark untuk disimpan."
        )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if output_file.exists():
        raise FileExistsError(
            f"File hasil sudah ada: {output_file}\n"
            "Pindahkan atau arsipkan file lama terlebih dahulu "
            "jika ingin membuat hasil baru."
        )

    for record in records:
        missing = set(CSV_COLUMNS) - set(record)

        if missing:
            raise ValueError(
                "Record memiliki kolom yang hilang: "
                f"{sorted(missing)}"
            )

    with output_file.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=CSV_COLUMNS,
        )
        writer.writeheader()
        writer.writerows(records)

    print(f"\nHasil benchmark disimpan: {output_file}")
    print(f"Jumlah record: {len(records)}")