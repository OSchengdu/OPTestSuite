import mxnet as mx
from mxnet import nd
from benchmark.opperf.utils.benchmark_utils import run_performance_test
import json

ctx = mx.cpu()
op = nd.slice_like

inputs = [{"data": (1024, 1024), "shape_like": (512, 512)}]

slice_like_res = run_performance_test(
    op,
    run_backward=True,
    dtype='float32',
    ctx=ctx,
    inputs=inputs,
    warmup=10,
    runs=25
)

with open('slice_like.json', 'w') as f:
    json.dump(slice_like_res, f, indent=4)

print(json.dumps(slice_like_res, indent=4))
