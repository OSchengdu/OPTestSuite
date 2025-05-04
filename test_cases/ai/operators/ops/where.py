import mxnet as mx
from mxnet import nd
from benchmark.opperf.utils.benchmark_utils import run_performance_test
import json

ctx = mx.cpu() 
op = nd.where
inputs = [{"condition": (1024, 1024), "x": (1024, 1024), "y": (1024, 1024)}]

where_res = run_performance_test(
    op,
    run_backward=True,
    dtype='float32',
    ctx=ctx,
    inputs=inputs,
    warmup=10,
    runs=25
)

with open('where.json', 'w') as f:
    json.dump(where_res, f, indent=4)

print(json.dumps(where_res, indent=4))
