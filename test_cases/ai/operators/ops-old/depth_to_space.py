import mxnet as mx
from mxnet import nd
from benchmark.opperf.utils.benchmark_utils import run_performance_test
import json

ctx = mx.cpu()
op = nd.depth_to_space

inputs = [{"data": (1, 1024, 1024), "block_size": 2}]

depth_to_space_res = run_performance_test(
    op,
    run_backward=True,
    dtype='float32',
    ctx=ctx,
    inputs=inputs,
    warmup=10,
    runs=25
)

with open('mxnet_operator_benchmark_results.json', 'w') as f:
    json.dump(depth_to_space_res, f, indent=4)

print(json.dumps(depth_to_space_res, indent=4))
