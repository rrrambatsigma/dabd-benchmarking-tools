FROM python:3.12-slim
WORKDIR /workspace
RUN pip install --no-cache-dir "dask[dataframe]" pyarrow psutil pandas
COPY benchmarks/dask/run_benchmark.py /workspace/run_benchmark.py
CMD ["python", "/workspace/run_benchmark.py"]