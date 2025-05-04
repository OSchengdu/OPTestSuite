import mxnet as mx
from mxnet import nd
from benchmark.opperf.utils.benchmark_utils import run_performance_test
import json

ctx = mx.cpu()
op = nd.slice_axis

inputs = [{"data": (1024, 1024), "axis": 0, "begin": 0, "end": 512}]

slice_axis_res = run_performance_test(
    op,
    run_backward=True,
    dtype='float32',
    ctx=ctx,
    inputs=inputs,
    warmup=10,
    runs=25
)

with open('slice_axis.json', 'w') as f:
    json.dump(slice_axis_res, f, indent=4)

print(json.dumps(slice_axis_res, indent=4))
