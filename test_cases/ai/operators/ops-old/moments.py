import mxnet as mx
from mxnet import nd
from benchmark.opperf.utils.benchmark_utils import run_performance_test
import json

ctx = mx.cpu()
op = nd.moments

inputs = [{"data": (1024, 1024), "axes": [0, 1]}]

moments_res = run_performance_test(
    op,
    run_backward=True,
    dtype='float32',
    ctx=ctx,
    inputs=inputs,
    warmup=10,
    runs=25
)

with open('moments.json', 'w') as f:
    json.dump(moments_res, f, indent=4)

print(json.dumps(moments_res, indent=4))
