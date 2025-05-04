import mxnet as mx
from mxnet import nd
from benchmark.opperf.utils.benchmark_utils import run_performance_test
import json

ctx = mx.cpu()
op = nd.Custom

data = nd.random.normal(shape=(1, 3, 256, 256), ctx=ctx)
inputs = [{"data": data, "op_type": "identity"}]

custom_res = run_performance_test(
    op,
    run_backward=True,
    dtype='float32',
    ctx=ctx,
    inputs=inputs,
    warmup=10,
    runs=25
)

with open('Custom.json', 'w') as f:
    json.dump(custom_res, f, indent=4)

print(json.dumps(custom_res, indent=4))
