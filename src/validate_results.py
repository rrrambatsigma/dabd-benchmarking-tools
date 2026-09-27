
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# KONFIGURASI PATH
# ============================================================

PROJECT_ROOT = Path("/workspace")

TOOLS = ["pandas", "polars", "dask"]

RESULTS_DIR = PROJECT_ROOT / "results" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "results" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "validation_report.csv"

EXPECTED_RECORDS_PER_TOOL = 60

EXPECTED_DATASETS = ["100k", "1m", "full"]

EXPECTED_OPERATIONS = [
    "read",
    "filter",
    "aggregation",
    "groupby",
]

EXPECTED_REPEATS = [1, 2, 3, 4, 5]

# Status yang dianggap berhasil
SUCCESS_STATUSES = {"success", "passed"}

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
# METRIK PEMBANDING BERDASARKAN OPERASI
# ============================================================

# Hanya membandingkan metrik yang relevan untuk setiap operasi.
#
# read:
#   - result_rows
#
# filter:
#   - result_rows
#
# aggregation:
#   - result_sum
#
# groupby:
#   - result_groups
#   - result_sum
#
# Catatan:
# result_rows tidak dibandingkan pada aggregation dan groupby
# karena nilainya dapat berupa NaN/tidak relevan.

OPERATION_METRICS = {
    "read": ["result_rows"],
    "filter": ["result_rows"],
    "aggregation": ["result_sum"],
    "groupby": ["result_groups", "result_sum"],
}


# ============================================================
# UTILITAS
# ============================================================

def load_tool_results(tool):
    """Membaca CSV hasil benchmark dari satu tool."""

    file_path = RESULTS_DIR / tool / "benchmark_results.csv"

    if not file_path.exists():
        raise FileNotFoundError(
            f"File hasil {tool} tidak ditemukan: {file_path}"
        )

    df = pd.read_csv(file_path)

    missing_columns = [
        col
        for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Kolom berikut tidak ditemukan pada {tool}: "
            f"{missing_columns}"
        )

    return df


def check_expected_experiments(df, tool):
    """Memeriksa kombinasi dataset, operasi, dan repeat."""

    expected = {
        (dataset, operation, repeat)
        for dataset in EXPECTED_DATASETS
        for operation in EXPECTED_OPERATIONS
        for repeat in EXPECTED_REPEATS
    }

    actual = set(
        zip(
            df["dataset"],
            df["operation"],
            df["repeat"],
        )
    )

    missing = expected - actual
    unexpected = actual - expected

    if missing:
        print(
            f"[WARNING] {tool}: "
            f"{len(missing)} eksperimen hilang."
        )

    if unexpected:
        print(
            f"[WARNING] {tool}: "
            f"{len(unexpected)} eksperimen "
            "tidak sesuai rancangan."
        )

    return len(missing), len(unexpected)


def check_duplicates(df, tool):
    """Memeriksa duplikasi identitas eksperimen."""

    keys = ["dataset", "operation", "repeat"]

    duplicate_count = int(
        df.duplicated(subset=keys, keep=False).sum()
    )

    if duplicate_count:
        print(
            f"[WARNING] {tool}: "
            f"{duplicate_count} baris terlibat dalam duplikasi."
        )

    return duplicate_count


def check_numeric_values(df, tool):
    """Memeriksa nilai numerik dan waktu eksekusi."""

    numeric_columns = [
        "dataset_size",
        "repeat",
        "wall_time_seconds",
        "cpu_time_seconds",
        "memory_mb",
    ]

    issues = []

    for column in numeric_columns:
        values = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        invalid = values.isna() | ~np.isfinite(values)

        if invalid.any():
            issues.append(
                f"{column}: "
                f"{int(invalid.sum())} nilai tidak valid"
            )

    for column in [
        "wall_time_seconds",
        "cpu_time_seconds",
    ]:
        values = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        invalid = values < 0

        if invalid.any():
            issues.append(
                f"{column}: "
                f"{int(invalid.sum())} nilai negatif"
            )

    memory_values = pd.to_numeric(
        df["memory_mb"],
        errors="coerce",
    )

    if (memory_values < 0).any():
        issues.append(
            "memory_mb: terdapat nilai negatif"
        )

    if issues:
        print(f"[WARNING] {tool}:")

        for issue in issues:
            print(f"  - {issue}")

    return issues


# ============================================================
# VALIDASI SATU TOOL
# ============================================================

def validate_single_tool(tool, df):
    """Memeriksa kelengkapan dan konsistensi hasil satu tool."""

    print(f"\nMemvalidasi: {tool}")

    issues = []

    # --------------------------------------------------------
    # 1. Validasi jumlah record
    # --------------------------------------------------------

    if len(df) != EXPECTED_RECORDS_PER_TOOL:
        issues.append(
            f"Jumlah record {len(df)}, "
            f"seharusnya {EXPECTED_RECORDS_PER_TOOL}"
        )

    # --------------------------------------------------------
    # 2. Validasi nama tool
    # --------------------------------------------------------

    if not (df["tool"] == tool).all():
        issues.append(
            "Kolom tool mengandung nama tool yang tidak sesuai"
        )

    # --------------------------------------------------------
    # 3. Validasi status benchmark
    # --------------------------------------------------------

    normalized_status = (
        df["validation_status"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    failed_status = ~normalized_status.isin(
        SUCCESS_STATUSES
    )

    failed_records = int(failed_status.sum())

    if failed_records:
        issues.append(
            f"{failed_records} record memiliki "
            "status selain passed/success"
        )

    # --------------------------------------------------------
    # 4. Validasi kombinasi eksperimen
    # --------------------------------------------------------

    missing, unexpected = check_expected_experiments(
        df,
        tool,
    )

    if missing:
        issues.append(
            f"{missing} kombinasi eksperimen hilang"
        )

    if unexpected:
        issues.append(
            f"{unexpected} kombinasi eksperimen tidak sesuai"
        )

    # --------------------------------------------------------
    # 5. Validasi duplikasi
    # --------------------------------------------------------

    duplicates = check_duplicates(df, tool)

    if duplicates:
        issues.append(
            f"{duplicates} baris duplikat ditemukan"
        )

    # --------------------------------------------------------
    # 6. Validasi nilai numerik
    # --------------------------------------------------------

    numeric_issues = check_numeric_values(df, tool)

    issues.extend(numeric_issues)

    # --------------------------------------------------------
    # 7. Status akhir
    # --------------------------------------------------------

    status = "success" if not issues else "failed"

    print(f"Status: {status}")

    for issue in issues:
        print(f"  - {issue}")

    return {
        "tool": tool,
        "total_records": len(df),
        "expected_records": EXPECTED_RECORDS_PER_TOOL,
        "failed_records": failed_records,
        "missing_experiments": missing,
        "unexpected_experiments": unexpected,
        "duplicate_rows": duplicates,
        "status": status,
        "issues": "; ".join(issues),
    }


# ============================================================
# VALIDASI KESETARAAN HASIL ANTAR-TOOLS
# ============================================================

def validate_cross_tool(results):
    """Membandingkan hasil operasi antar-tools."""

    print("\n" + "=" * 70)
    print("VALIDASI KESETARAAN HASIL ANTAR-TOOLS")
    print("=" * 70)

    combined = pd.concat(
        results.values(),
        ignore_index=True,
    )

    comparison_keys = [
        "dataset",
        "operation",
        "repeat",
    ]

    report = []

    for key, group in combined.groupby(
        comparison_keys,
        dropna=False,
    ):
        dataset, operation, repeat = key

        row = {
            "dataset": dataset,
            "operation": operation,
            "repeat": repeat,
            "tools_present": group["tool"].nunique(),
            "status": "success",
            "details": "",
        }

        # ----------------------------------------------------
        # 1. Periksa kelengkapan tools
        # ----------------------------------------------------

        if group["tool"].nunique() != len(TOOLS):
            row["status"] = "failed"
            row["details"] = (
                "Jumlah tools tidak lengkap"
            )

            report.append(row)
            continue

        details = []

        # ----------------------------------------------------
        # 2. Tentukan metrik sesuai operasi
        # ----------------------------------------------------

        metrics = OPERATION_METRICS.get(operation)

        if metrics is None:
            row["status"] = "failed"
            row["details"] = (
                f"Operasi tidak dikenal: {operation}"
            )

            report.append(row)
            continue

        # ----------------------------------------------------
        # 3. Bandingkan metrik antar-tools
        # ----------------------------------------------------

        for column in metrics:
            values = pd.to_numeric(
                group[column],
                errors="coerce",
            )

            # Jika metrik yang wajib dibandingkan kosong
            # pada seluruh tools, hasil tidak dapat diverifikasi.
            if values.isna().all():
                details.append(
                    f"{column}: seluruh nilai kosong"
                )
                continue

            # Jika sebagian nilai kosong, perbandingan
            # tidak dapat dilakukan secara lengkap.
            if values.isna().any():
                details.append(
                    f"{column}: ada nilai kosong"
                )
                continue

            numeric_values = values.to_numpy(
                dtype=float
            )

            # result_sum dapat berupa angka desimal.
            if column == "result_sum":
                is_equal = np.allclose(
                    numeric_values,
                    numeric_values[0],
                    rtol=1e-6,
                    atol=1e-6,
                    equal_nan=False,
                )

            # result_rows dan result_groups berupa jumlah.
            else:
                is_equal = (
                    values.nunique() == 1
                )

            if not is_equal:
                details.append(
                    f"{column}: hasil berbeda antar-tools"
                )

        # ----------------------------------------------------
        # 4. Tentukan status kombinasi
        # ----------------------------------------------------

        if details:
            row["status"] = "failed"
            row["details"] = "; ".join(details)

        report.append(row)

    report_df = pd.DataFrame(report)

    failed_count = int(
        (report_df["status"] == "failed").sum()
    )

    success_count = int(
        (report_df["status"] == "success").sum()
    )

    print(
        f"Total kombinasi dibandingkan: "
        f"{len(report_df)}"
    )

    print(
        f"Kombinasi berhasil: {success_count}"
    )

    print(
        f"Kombinasi gagal: {failed_count}"
    )

    return report_df


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("VALIDASI HASIL BENCHMARK")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = {}
    tool_reports = []

    # --------------------------------------------------------
    # 1. Validasi masing-masing tool
    # --------------------------------------------------------

    for tool in TOOLS:
        try:
            df = load_tool_results(tool)

            results[tool] = df

            tool_report = validate_single_tool(
                tool,
                df,
            )

            tool_reports.append(tool_report)

        except Exception as error:
            print(
                f"\n[ERROR] {tool}: {error}"
            )

            tool_reports.append({
                "tool": tool,
                "total_records": 0,
                "expected_records": EXPECTED_RECORDS_PER_TOOL,
                "failed_records": 0,
                "missing_experiments": None,
                "unexpected_experiments": None,
                "duplicate_rows": None,
                "status": "failed",
                "issues": str(error),
            })

    # --------------------------------------------------------
    # 2. Simpan laporan validasi per tool
    # --------------------------------------------------------

    tool_report_df = pd.DataFrame(
        tool_reports
    )

    tool_report_path = (
        OUTPUT_DIR / "tool_validation_report.csv"
    )

    tool_report_df.to_csv(
        tool_report_path,
        index=False,
    )

    # --------------------------------------------------------
    # 3. Validasi kesetaraan antar-tools
    # --------------------------------------------------------

    if len(results) == len(TOOLS):
        cross_report = validate_cross_tool(
            results
        )

        cross_report.to_csv(
            OUTPUT_FILE,
            index=False,
        )

        print(
            f"\nLaporan perbandingan: {OUTPUT_FILE}"
        )

    else:
        print(
            "\n[WARNING] Validasi antar-tools dilewati "
            "karena ada file hasil benchmark "
            "yang belum tersedia."
        )

    # --------------------------------------------------------
    # 4. Informasi laporan
    # --------------------------------------------------------

    print(
        f"Laporan validasi tools: "
        f"{tool_report_path}"
    )

    print("\n" + "=" * 70)
    print("VALIDASI SELESAI")
    print("=" * 70)


if __name__ == "__main__":
    main()