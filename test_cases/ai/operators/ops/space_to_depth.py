import mxnet as mx
from mxnet import nd
from benchmark.opperf.utils.benchmark_utils import run_performance_test
import json

ctx = mx.cpu()
op = nd.space_to_depth

inputs = [{"data": (1, 1024, 1024), "block_size": 2}]

space_to_depth_res = run_performance_test(
    op,
    run_backward=True,
    dtype='float32',
    ctx=ctx,
    inputs=inputs,
    warmup=10,
    runs=25
)

with open('mxnet_operator_benchmark_results.json', 'w') as f:
    json.dump(space_to_depth_res, f, indent=4)

print(json.dumps(space_to_depth_res, indent=4))
