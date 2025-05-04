import mxnet as mx
from mxnet import nd
from benchmark.opperf.utils.benchmark_utils import run_performance_test
import json

ctx = mx.cpu()
op = nd.reshape_like

inputs = [{"lhs": (1024, 1024), "rhs": (1024 * 1024,)}]

reshape_like_res = run_performance_test(
    op,
    run_backward=True,
    dtype='float32',
    ctx=ctx,
    inputs=inputs,
    warmup=10,
    runs=25
)
with open('mxnet_operator_benchmark_results.json', 'w') as f:
    json.dump(reshape_like_res, f, indent=4)

print(json.dumps(reshape_like_res, indent=4))
