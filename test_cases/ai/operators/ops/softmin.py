import mxnet as mx
from mxnet import nd
from benchmark.opperf.utils.benchmark_utils import run_performance_test
import json

ctx = mx.cpu()
op = nd.softmin

data = nd.random.normal(shape=(1024, 10), ctx=ctx)
inputs = [{"data": data, "axis": 1}]

softmin_res = run_performance_test(
    op,
    run_backward=True,
    dtype='float32',
    ctx=ctx,
    inputs=inputs,
    warmup=10,
    runs=25
)

with open('softmin.json', 'w') as f:
    json.dump(softmin_res, f, indent=4)

print(json.dumps(softmin_res, indent=4))
