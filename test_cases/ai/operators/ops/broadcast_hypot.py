import mxnet as mx
from mxnet import nd
from benchmark.opperf.utils.benchmark_utils import run_performance_test
import json

ctx = mx.cpu()
op = nd.broadcast_hypot

inputs = [{"lhs": (1024, 1024), "rhs": (1024, 1024)}]
broadcast_hypot_res = run_performance_test(
    op,
    run_backward=True,
    dtype='float32',
    ctx=ctx,
    inputs=inputs,
    warmup=10,
    runs=25
)

with open('broadcast_hypot.json', 'w') as f:
    json.dump(broadcast_hypot_res, f, indent=4)

print(json.dumps(broadcast_hypot_res, indent=4))
