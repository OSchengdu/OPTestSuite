import mxnet as mx
from mxnet import nd
from benchmark.opperf.utils.benchmark_utils import run_performance_test
import json

ctx = mx.cpu() 
op = nd.broadcast_greater_equal

inputs = [{"lhs": (1024, 1024), "rhs": (1024, 1024)}]

broadcast_greater_equal_res = run_performance_test(
    op,
    run_backward=True,
    dtype='float32',
    ctx=ctx,
    inputs=inputs,
    warmup=10,
    runs=25
)

with open('broadcast_greater_equal.json', 'w') as f:
    json.dump(broadcast_greater_equal_res, f, indent=4)

print(json.dumps(broadcast_greater_equal_res, indent=4))
