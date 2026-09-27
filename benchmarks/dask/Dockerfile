
FROM python:3.12-slim

WORKDIR /workspace

RUN pip install --no-cache-dir "dask[dataframe]" pandas pyarrow psutil

COPY src/ /workspace/src/
COPY benchmarks/dask/run_benchmark.py /workspace/benchmarks/dask/run_benchmark.py

CMD ["python", "/workspace/benchmarks/dask/run_benchmark.py"]