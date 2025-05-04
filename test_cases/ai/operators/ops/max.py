import mxnet as mx
from mxnet import nd
from benchmark.opperf.utils.benchmark_utils import run_performance_test
import json

ctx = mx.cpu()
op = nd.max

inputs = [{"data": nd.random.normal(shape=(1024, 1024), ctx=ctx)}]
max_res = run_performance_test(
    op,
    run_backward=True,
    dtype='float32',
    ctx=ctx,
    inputs=inputs,
    warmup=10,
    runs=25
)

with open('max.json', 'w') as f:
    json.dump(max_res, f, indent=4)

print(json.dumps(max_res, indent=4))
