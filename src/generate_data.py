"""Create reproducible row-count samples from the source Parquet file."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "raw" / "yellow_tripdata_2026-07.parquet"
OUTPUT = ROOT / "data" / "processed"
SAMPLES = {"sample_100k.parquet": 100_000, "sample_1m.parquet": 1_000_000, "sample_5m.parquet": 5_000_000}

def main():
    if not SOURCE.is_file():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {SOURCE}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(SOURCE)
    for name, n in SAMPLES.items():
        sample = frame.head(n)
        target = OUTPUT / name
        sample.to_parquet(target, index=False)
        print(f"{target}: {len(sample):,} rows")

if __name__ == "__main__":
    main()