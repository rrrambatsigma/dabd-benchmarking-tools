"""Check that benchmark CSV outputs exist and contain expected columns."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FILES = [ROOT / "results" / "raw" / tool / "benchmark_results.csv" for tool in ("pandas", "polars", "dask")]
REQUIRED = {"tool", "operation", "dataset_size", "repeat", "runtime_seconds", "validated"}

def main():
    for path in FILES:
        if not path.is_file() or path.stat().st_size == 0:
            print(f"MISSING: {path}")
            continue
        frame = pd.read_csv(path)
        missing = REQUIRED - set(frame.columns)
        print(f"{'OK' if not missing else 'INVALID'}: {path} rows={len(frame)} missing={sorted(missing)}")

if __name__ == "__main__":
    main()