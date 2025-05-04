import mxnet as mx
from mxnet import nd
from benchmark.opperf.utils.benchmark_utils import run_performance_test
import json

ctx = mx.cpu() 
op = nd.swapaxes

inputs = [{"data": (1024, 1024), "dim1": 0, "dim2": 1}]

swapaxes_res = run_performance_test(
    op,
    run_backward=True,
    dtype='float32',
    ctx=ctx,
    inputs=inputs,
    warmup=10,
    runs=25
)

with open('mxnet_operator_benchmark_results.json', 'w') as f:
    json.dump(swapaxes_res, f, indent=4)
print(json.dumps(swapaxes_res, indent=4))
