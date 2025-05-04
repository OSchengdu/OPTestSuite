import mxnet as mx
from mxnet import nd
from benchmark.opperf.utils.benchmark_utils import run_performance_test
import json

ctx = mx.cpu() 
op = nd.gather_nd

inputs = [{"data": (1024, 1024), "indices": (1024, 2)}]

gather_nd_res = run_performance_test(
    op,
    run_backward=True,
    dtype='float32',
    ctx=ctx,
    inputs=inputs,
    warmup=10,
    runs=25
)

with open('gather_nd.json', 'w') as f:
    json.dump(gather_nd_res, f, indent=4)

print(json.dumps(gather_nd_res, indent=4))
