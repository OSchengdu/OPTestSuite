import mxnet as mx
from mxnet import nd
from benchmark.opperf.utils.benchmark_utils import run_performance_test
import json

ctx = mx.cpu() 
op = nd.slice

inputs = [{"data": (1024, 1024), "begin": (0, 0), "end": (512, 512)}]

slice_res = run_performance_test(
    op,
    run_backward=True,
    dtype='float32',
    ctx=ctx,
    inputs=inputs,
    warmup=10,
    runs=25
)

with open('slice.json', 'w') as f:
    json.dump(slice_res, f, indent=4)

print(json.dumps(slice_res, indent=4))
