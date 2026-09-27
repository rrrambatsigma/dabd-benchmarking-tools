
from pathlib import Path

import pandas as pd


# ============================================================
# KONFIGURASI PATH
# ============================================================

PROJECT_ROOT = Path("/workspace")

TOOLS = ["pandas", "polars", "dask"]

RESULTS_DIR = PROJECT_ROOT / "results" / "raw"

OUTPUT_DIR = PROJECT_ROOT / "results" / "processed"

MERGED_FILE = OUTPUT_DIR / "merged_results.csv"

SUMMARY_FILE = OUTPUT_DIR / "benchmark_summary.csv"


# ============================================================
# KOLOM WAJIB
# ============================================================

REQUIRED_COLUMNS = [
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
# STATUS VALIDASI
# ============================================================

# Kedua status berikut dianggap berhasil.
SUCCESS_STATUSES = {"success", "passed"}


# ============================================================
# MEMBACA HASIL BENCHMARK
# ============================================================

def load_results(tool):
    """Membaca file CSV hasil benchmark dari satu tool."""

    file_path = (
        RESULTS_DIR
        / tool
        / "benchmark_results.csv"
    )

    if not file_path.exists():
        raise FileNotFoundError(
            f"File hasil {tool} tidak ditemukan: {file_path}"
        )

    df = pd.read_csv(file_path)

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Kolom wajib tidak ditemukan pada {tool}: "
            f"{missing_columns}"
        )

    print(
        f"[OK] {tool}: {len(df)} record berhasil dibaca."
    )

    return df


# ============================================================
# MERGE HASIL
# ============================================================

def merge_results():
    """Menggabungkan hasil benchmark seluruh tools."""

    print("=" * 70)
    print("MERGE HASIL BENCHMARK")
    print("=" * 70)

    dataframes = []

    for tool in TOOLS:
        df = load_results(tool)

        # Memastikan nama tool konsisten dengan sumber file.
        df["tool"] = tool

        dataframes.append(df)

    # --------------------------------------------------------
    # Gabungkan seluruh hasil benchmark
    # --------------------------------------------------------

    merged_df = pd.concat(
        dataframes,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Mengurutkan hasil agar mudah dibaca dan dianalisis
    # --------------------------------------------------------

    tool_order = {
        tool: index
        for index, tool in enumerate(TOOLS)
    }

    merged_df["_tool_order"] = (
        merged_df["tool"].map(tool_order)
    )

    dataset_order = {
        "100k": 0,
        "1m": 1,
        "full": 2,
    }

    merged_df["_dataset_order"] = (
        merged_df["dataset"].map(dataset_order)
    )

    operation_order = {
        "read": 0,
        "filter": 1,
        "aggregation": 2,
        "groupby": 3,
    }

    merged_df["_operation_order"] = (
        merged_df["operation"].map(operation_order)
    )

    merged_df = (
        merged_df.sort_values(
            by=[
                "_tool_order",
                "_dataset_order",
                "_operation_order",
                "repeat",
            ]
        )
        .drop(
            columns=[
                "_tool_order",
                "_dataset_order",
                "_operation_order",
            ]
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Simpan hasil gabungan
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    merged_df.to_csv(
        MERGED_FILE,
        index=False,
    )

    print("\n" + "=" * 70)
    print("MERGE SELESAI")
    print("=" * 70)

    print(f"Total tools : {len(TOOLS)}")
    print(f"Total record: {len(merged_df)}")
    print(f"File output : {MERGED_FILE}")

    return merged_df


# ============================================================
# MEMBUAT RINGKASAN STATISTIK
# ============================================================

def create_summary(merged_df):
    """Menghitung statistik ringkasan benchmark."""

    print("\n" + "=" * 70)
    print("MEMBUAT RINGKASAN BENCHMARK")
    print("=" * 70)

    # --------------------------------------------------------
    # Kolom pengelompokan
    # --------------------------------------------------------

    group_columns = [
        "tool",
        "dataset",
        "dataset_size",
        "operation",
    ]

    # --------------------------------------------------------
    # Kolom metrik yang dihitung
    # --------------------------------------------------------

    metric_columns = [
        "wall_time_seconds",
        "cpu_time_seconds",
        "memory_mb",
    ]

    # Pastikan metrik berbentuk numerik.
    for column in metric_columns:
        merged_df[column] = pd.to_numeric(
            merged_df[column],
            errors="coerce",
        )

    # --------------------------------------------------------
    # Hitung mean dan standar deviasi
    # --------------------------------------------------------

    grouped = merged_df.groupby(
        group_columns,
        dropna=False,
    )

    summary_mean = (
        grouped[metric_columns]
        .mean()
        .add_suffix("_mean")
    )

    summary_std = (
        grouped[metric_columns]
        .std(ddof=1)
        .add_suffix("_std")
    )

    summary = pd.concat(
        [
            summary_mean,
            summary_std,
        ],
        axis=1,
    ).reset_index()

    # --------------------------------------------------------
    # Jumlah pengulangan yang tersedia
    # --------------------------------------------------------

    repeat_count = (
        grouped["repeat"]
        .count()
        .rename("repeat_count")
        .reset_index()
    )

    summary = summary.merge(
        repeat_count,
        on=group_columns,
        how="left",
    )

    # --------------------------------------------------------
    # Normalisasi status validasi
    # --------------------------------------------------------

    validation_status = (
        merged_df["validation_status"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # passed dan success dianggap berhasil.
    merged_df["_is_success"] = (
        validation_status.isin(SUCCESS_STATUSES)
    )

    # --------------------------------------------------------
    # Hitung jumlah record berhasil dan gagal
    # --------------------------------------------------------

    validation_summary = (
        merged_df.groupby(
            group_columns,
            dropna=False,
        )["_is_success"]
        .agg(
            success_count="sum",
            failed_count=lambda values: int(
                (~values).sum()
            ),
        )
        .reset_index()
    )

    summary = summary.merge(
        validation_summary,
        on=group_columns,
        how="left",
    )

    # --------------------------------------------------------
    # Pastikan jumlah status berupa integer
    # --------------------------------------------------------

    summary["success_count"] = (
        summary["success_count"].fillna(0).astype(int)
    )

    summary["failed_count"] = (
        summary["failed_count"].fillna(0).astype(int)
    )

    summary["repeat_count"] = (
        summary["repeat_count"].fillna(0).astype(int)
    )

    # --------------------------------------------------------
    # Urutkan ringkasan
    # --------------------------------------------------------

    tool_order = {
        tool: index
        for index, tool in enumerate(TOOLS)
    }

    dataset_order = {
        "100k": 0,
        "1m": 1,
        "full": 2,
    }

    operation_order = {
        "read": 0,
        "filter": 1,
        "aggregation": 2,
        "groupby": 3,
    }

    summary["_tool_order"] = (
        summary["tool"].map(tool_order)
    )

    summary["_dataset_order"] = (
        summary["dataset"].map(dataset_order)
    )

    summary["_operation_order"] = (
        summary["operation"].map(operation_order)
    )

    summary = (
        summary.sort_values(
            by=[
                "_tool_order",
                "_dataset_order",
                "_operation_order",
            ]
        )
        .drop(
            columns=[
                "_tool_order",
                "_dataset_order",
                "_operation_order",
            ]
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Simpan ringkasan
    # --------------------------------------------------------

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    print(f"File ringkasan: {SUMMARY_FILE}")
    print(f"Jumlah kelompok: {len(summary)}")

    return summary


# ============================================================
# MAIN
# ============================================================

def main():
    """Menjalankan proses merge dan pembuatan ringkasan."""

    merged_df = merge_results()

    summary_df = create_summary(
        merged_df
    )

    print("\n" + "=" * 70)
    print("PROSES MERGE DAN SUMMARY SELESAI")
    print("=" * 70)

    print(
        f"Total record gabungan: {len(merged_df)}"
    )

    print(
        f"Total baris ringkasan: {len(summary_df)}"
    )

    print("\nPreview ringkasan:")

    print(
        summary_df.to_string(index=False)
    )


if __name__ == "__main__":
    main()