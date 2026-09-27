FROM python:3.12-slim

WORKDIR /workspace

RUN pip install --no-cache-dir pandas pyarrow psutil

COPY src/ /workspace/src/
COPY benchmarks/pandas/run_benchmark.py /workspace/benchmarks/pandas/run_benchmark.py

CMD ["python", "/workspace/benchmarks/pandas/run_benchmark.py"]