import mxnet as mx
from mxnet import nd
from benchmark.opperf.utils.benchmark_utils import run_performance_test
import json

ctx = mx.cpu()
op = nd.squeeze

data = nd.random.normal(shape=(1, 3, 1, 256), ctx=ctx)
inputs = [{"data": data, "axis": None}]

squeeze_res = run_performance_test(
    op,
    run_backward=True,
    dtype='float32',
    ctx=ctx,
    inputs=inputs,
    warmup=10,
    runs=25
)

with open('squeeze.json', 'w') as f:
    json.dump(squeeze_res, f, indent=4)

print(json.dumps(squeeze_res, indent=4))
