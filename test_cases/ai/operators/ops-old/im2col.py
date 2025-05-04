import mxnet as mx
from mxnet import nd
from benchmark.opperf.utils.benchmark_utils import run_performance_test
import json

ctx = mx.cpu()
op = nd.im2col

data = nd.random.normal(shape=(1, 3, 32, 32), ctx=ctx)
inputs = [{"data": data, "kernel": (3, 3), "stride": (1, 1), "dilate": (1, 1), "pad": (0, 0)}]

im2col_res = run_performance_test(
    op,
    run_backward=True,
    dtype='float32',
    ctx=ctx,
    inputs=inputs,
    warmup=10,
    runs=25
)

with open('im2col.json', 'w') as f:
    json.dump(im2col_res, f, indent=4)

print(json.dumps(im2col_res, indent=4))
