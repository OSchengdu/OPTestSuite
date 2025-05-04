import mxnet as mx
from mxnet import nd
from benchmark.opperf.utils.benchmark_utils import run_performance_test
import json

ctx = mx.cpu()
op = nd.broadcast_power

inputs = [{"lhs": (1024, 1024), "rhs": (1024, 1024)}]

broadcast_power_res = run_performance_test(
    op,
    run_backward=True,
    dtype='float32',
    ctx=ctx,
    inputs=inputs,
    warmup=10,
    runs=25
)

with open('broadcast_power.json', 'w') as f:
    json.dump(broadcast_power_res, f, indent=4)

print(json.dumps(broadcast_power_res, indent=4))
