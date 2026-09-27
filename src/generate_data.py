
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


# ============================================================
# KONFIGURASI PATH DAN DATASET
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

SOURCE_FILE = RAW_DIR / "yellow_tripdata_2026-07.parquet"

DATASETS = {
    "taxi_100k.parquet": 100_000,
    "taxi_1m.parquet": 1_000_000,
    "taxi_full.parquet": None,
}

DEFAULT_BATCH_SIZE = 100_000


# ============================================================
# VALIDASI SUMBER
# ============================================================

def validate_source(source_file: Path) -> pq.ParquetFile:
    """Memastikan file sumber tersedia dan dapat dibaca."""

    if not source_file.exists():
        raise FileNotFoundError(
            f"Dataset sumber tidak ditemukan: {source_file}"
        )

    if not source_file.is_file():
        raise ValueError(
            f"Path sumber bukan file: {source_file}"
        )

    parquet_file = pq.ParquetFile(source_file)

    if parquet_file.metadata is None:
        raise ValueError(
            "Metadata Parquet tidak ditemukan."
        )

    if parquet_file.metadata.num_rows <= 0:
        raise ValueError(
            "Dataset sumber tidak memiliki baris data."
        )

    schema = parquet_file.schema_arrow

    required_columns = {"trip_distance", "PULocationID"}
    available_columns = set(schema.names)
    missing_columns = required_columns - available_columns

    if missing_columns:
        raise ValueError(
            f"Kolom wajib tidak ditemukan: {sorted(missing_columns)}"
        )

    print("\n=== INFORMASI DATASET SUMBER ===")
    print(f"File       : {source_file}")
    print(f"Jumlah baris: {parquet_file.metadata.num_rows:,}")
    print(f"Jumlah kolom: {len(schema.names)}")
    print(f"Row groups : {parquet_file.metadata.num_row_groups}")
    print("\nSkema kolom:")
    print(schema)

    return parquet_file


# ============================================================
# PEMBENTUKAN SUBSET BERDASARKAN URUTAN AWAL
# ============================================================

def create_prefix_dataset(
    source_file: Path,
    output_file: Path,
    target_rows: int,
    batch_size: int,
) -> int:
    """
    Membuat subset awal sebanyak target_rows baris.

    Baris diproses sesuai urutan pembacaan Parquet.
    Tidak ada sampling acak atau transformasi nilai.
    """

    parquet_file = pq.ParquetFile(source_file)
    source_rows = parquet_file.metadata.num_rows

    if target_rows > source_rows:
        raise ValueError(
            f"Jumlah baris target ({target_rows:,}) melebihi "
            f"jumlah baris sumber ({source_rows:,})."
        )

    schema = parquet_file.schema_arrow
    rows_written = 0

    with pq.ParquetWriter(
        output_file,
        schema=schema,
        compression="snappy",
    ) as writer:

        for batch in parquet_file.iter_batches(
            batch_size=batch_size
        ):
            remaining = target_rows - rows_written

            if remaining <= 0:
                break

            selected_batch = batch.slice(
                offset=0,
                length=min(batch.num_rows, remaining),
            )

            if selected_batch.num_rows > 0:
                writer.write_batch(selected_batch)
                rows_written += selected_batch.num_rows

            if rows_written >= target_rows:
                break

    if rows_written != target_rows:
        raise RuntimeError(
            f"Jumlah baris hasil tidak sesuai. "
            f"Target={target_rows:,}, hasil={rows_written:,}"
        )

    return rows_written


# ============================================================
# VALIDASI FILE HASIL
# ============================================================

def validate_output(
    output_file: Path,
    expected_rows: int,
    expected_schema: pa.Schema,
) -> None:
    """Memeriksa jumlah baris dan skema file hasil."""

    if not output_file.exists():
        raise FileNotFoundError(
            f"File hasil tidak ditemukan: {output_file}"
        )

    result = pq.ParquetFile(output_file)

    actual_rows = result.metadata.num_rows
    actual_schema = result.schema_arrow

    if actual_rows != expected_rows:
        raise ValueError(
            f"Jumlah baris tidak sesuai pada {output_file.name}. "
            f"Target={expected_rows:,}, aktual={actual_rows:,}"
        )

    if not actual_schema.equals(
        expected_schema,
        check_metadata=True,
    ):
        raise ValueError(
            f"Skema hasil berbeda dari sumber pada {output_file.name}."
        )

    print(
        f"[VALID] {output_file.name}: "
        f"{actual_rows:,} baris, "
        f"{len(actual_schema.names)} kolom"
    )


# ============================================================
# PROSES UTAMA
# ============================================================

def generate_datasets(
    overwrite: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> None:
    """Membuat seluruh dataset preprocessing."""

    if batch_size <= 0:
        raise ValueError("batch_size harus lebih besar dari 0.")

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    source = validate_source(SOURCE_FILE)
    source_rows = source.metadata.num_rows
    source_schema = source.schema_arrow

    # Tentukan jumlah baris setiap dataset.
    expected_rows = {
        filename: (
            source_rows if target is None else target
        )
        for filename, target in DATASETS.items()
    }

    # Validasi semua target sebelum membuat file.
    for filename, target in DATASETS.items():
        if target is not None and target > source_rows:
            raise ValueError(
                f"Dataset {filename} memerlukan {target:,} baris, "
                f"sedangkan sumber hanya memiliki {source_rows:,} baris."
            )

    # Jangan menimpa file yang sudah ada tanpa izin eksplisit.
    existing_files = [
        PROCESSED_DIR / filename
        for filename in DATASETS
        if (PROCESSED_DIR / filename).exists()
    ]

    if existing_files and not overwrite:
        names = "\n".join(
            f"  - {path.name}" for path in existing_files
        )

        raise FileExistsError(
            "File hasil berikut sudah tersedia:\n"
            f"{names}\n"
            "Gunakan --overwrite jika memang ingin menggantinya."
        )

    print("\n=== MULAI PREPROCESSING ===")

    for filename, target in DATASETS.items():
        output_file = PROCESSED_DIR / filename
        expected = expected_rows[filename]

        print(f"\nMemproses: {filename}")

        if target is None:
            # Dataset full merupakan salinan utuh file sumber.
            shutil.copy2(SOURCE_FILE, output_file)
            print("Metode     : salin file sumber secara utuh")
        else:
            # Dataset subset diambil dari baris awal secara berurutan.
            written = create_prefix_dataset(
                source_file=SOURCE_FILE,
                output_file=output_file,
                target_rows=target,
                batch_size=batch_size,
            )
            print(f"Baris ditulis: {written:,}")

        validate_output(
            output_file=output_file,
            expected_rows=expected,
            expected_schema=source_schema,
        )

    print("\n=== PREPROCESSING SELESAI ===")
    print(f"Folder hasil: {PROCESSED_DIR}")


# ============================================================
# COMMAND-LINE INTERFACE
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Membuat dataset preprocessing NYC Yellow Taxi "
            "dari file Parquet sumber."
        )
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Izinkan penimpaan file hasil yang sudah tersedia.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=(
            "Jumlah baris per batch pembacaan. "
            f"Default: {DEFAULT_BATCH_SIZE:,}."
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        generate_datasets(
            overwrite=args.overwrite,
            batch_size=args.batch_size,
        )
        return 0

    except Exception as error:
        print(
            f"\n[ERROR] {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())