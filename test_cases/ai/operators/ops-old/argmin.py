import mxnet as mx
from mxnet import nd
from benchmark.opperf.utils.benchmark_utils import run_performance_test
import json

ctx = mx.cpu()
op = nd.argmin

inputs = [{"data": nd.random.normal(shape=(1024, 1024), ctx=ctx)}]
argmin_res = run_performance_test(
    op,
    run_backward=True,
    dtype='float32',
    ctx=ctx,
    inputs=inputs,
    warmup=10,
    runs=25
)

with open('argmin.json', 'w') as f:
    json.dump(argmin_res, f, indent=4)

print(json.dumps(argmin_res, indent=4))
