import mxnet as mx
from mxnet import nd
from benchmark.opperf.utils.benchmark_utils import run_performance_test
import json

ctx = mx.cpu()
op = nd.col2im

data = nd.random.normal(shape=(1, 3*3*3, 32*32), ctx=ctx)
inputs = [{"data": data, "output_shape": (1, 3, 32, 32), "kernel": (3, 3), "stride": (1, 1), "dilate": (1, 1), "pad": (0, 0)}]

col2im_res = run_performance_test(
    op,
    run_backward=True,
    dtype='float32',
    ctx=ctx,
    inputs=inputs,
    warmup=10,
    runs=25
)

with open('col2im.json', 'w') as f:
    json.dump(col2im_res, f, indent=4)

print(json.dumps(col2im_res, indent=4))
