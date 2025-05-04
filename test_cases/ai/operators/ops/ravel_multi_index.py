import mxnet as mx
from mxnet import nd
from benchmark.opperf.utils.benchmark_utils import run_performance_test
import json

ctx = mx.cpu() 
op = nd.ravel_multi_index

inputs = [{"data": (1024, 2), "shape": (1024, 1024)}]
ravel_multi_index_res = run_performance_test(
    op,
    run_backward=True,
    dtype='float32',
    ctx=ctx,
    inputs=inputs,
    warmup=10,
    runs=25
)

with open('ravel_multi_index.json', 'w') as f:
    json.dump(ravel_multi_index_res, f, indent=4)

print(json.dumps(ravel_multi_index_res, indent=4))
