"""Merge per-tool CSVs and create a runtime summary."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw"
OUT = ROOT / "results" / "processed"

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    paths = [RAW / tool / "benchmark_results.csv" for tool in ("pandas", "polars", "dask")]
    frames = [pd.read_csv(p) for p in paths if p.is_file() and p.stat().st_size > 0]
    if not frames:
        raise RuntimeError("Belum ada CSV hasil benchmark.")
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(OUT / "benchmark_summary.csv", index=False)
    keys = ["tool", "operation", "dataset_size"]
    summary = combined.groupby(keys, dropna=False)["runtime_seconds"].agg(["mean", "std", "min", "max"]).reset_index()
    summary.to_csv(OUT / "benchmark_comparison.csv", index=False)
    print(f"Hasil disimpan di {OUT}")

if __name__ == "__main__":
    main()