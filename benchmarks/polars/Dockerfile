
FROM python:3.12-slim

WORKDIR /workspace

RUN pip install --no-cache-dir polars pyarrow psutil

COPY src/ /workspace/src/

COPY benchmarks/polars/run_benchmark.py /workspace/benchmarks/polars/run_benchmark.py

CMD ["python", "/workspace/benchmarks/polars/run_benchmark.py"]