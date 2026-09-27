FROM python:3.12-slim
WORKDIR /workspace
RUN pip install --no-cache-dir pandas pyarrow psutil
COPY benchmarks/pandas/run_benchmark.py /workspace/run_benchmark.py
CMD ["python", "/workspace/run_benchmark.py"]