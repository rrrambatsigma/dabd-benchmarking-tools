# DABD Benchmarking Tools

Perbandingan pemrosesan dataset NYC Yellow Taxi menggunakan Pandas, Polars, dan Dask dalam container Docker terpisah. JupyterLab digunakan hanya untuk analisis CSV hasil.

## Struktur
- `benchmarks/`: Dockerfile dan runner tiap tool.
- `analysis/`: image dan dependencies Jupyter.
- `data/raw/`: dataset Parquet asli.
- `data/processed/`: dataset sampel.
- `results/raw/`: CSV mentah per tool.
- `results/processed/`: hasil gabungan dan ringkasan.
- `src/`: utilitas persiapan data, validasi, dan penggabungan.

## Dataset
Letakkan file bernama `yellow_tripdata_2026-07.parquet` di `data/raw/`.

## Validasi dan build
```powershell
docker compose config --quiet
docker compose build
```

## Jalankan benchmark satu per satu
```powershell
docker compose --profile benchmark run --rm pandas-benchmark
docker compose --profile benchmark run --rm polars-benchmark
docker compose --profile benchmark run --rm dask-benchmark
```

## JupyterLab
```powershell
docker compose up -d jupyter-analysis
docker compose logs jupyter-analysis
```
Buka `http://localhost:8888/lab` dan gunakan token dari log.

## Catatan
Jalankan benchmark secara berurutan agar pembatasan resource dan beban sistem konsisten. Skrip awal ini adalah kerangka implementasi; validasi metodologi, kolom dataset, dan kesetaraan hasil perlu dilakukan sebelum dipakai sebagai hasil penelitian.