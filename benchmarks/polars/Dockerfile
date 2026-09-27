FROM python:3.12-slim
WORKDIR /workspace
RUN pip install --no-cache-dir polars pyarrow psutil
COPY benchmarks/polars/run_benchmark.py /workspace/run_benchmark.py
CMD ["python", "/workspace/run_benchmark.py"]