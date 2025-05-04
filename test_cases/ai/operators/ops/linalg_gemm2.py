import mxnet as mx
from mxnet import nd
from benchmark.opperf.utils.benchmark_utils import run_performance_test
import json

ctx = mx.cpu()
op = nd.linalg_gemm2

inputs = [{"A": (1024, 1024), "B": (1024, 1024)}]

linalg_gemm2_res = run_performance_test(
    op,
    run_backward=True,
    dtype='float32',
    ctx=ctx,
    inputs=inputs,
    warmup=10,
    runs=25
)

with open('linalg_gemm2.json', 'w') as f:
    json.dump(linalg_gemm2_res, f, indent=4)

print(json.dumps(linalg_gemm2_res, indent=4))
