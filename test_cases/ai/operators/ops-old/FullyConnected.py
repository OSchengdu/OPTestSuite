import mxnet as mx
from mxnet import nd
from benchmark.opperf.utils.benchmark_utils import run_performance_test
import json

ctx = mx.cpu()op = nd.FullyConnected

data = nd.random.normal(shape=(1024, 1024), ctx=ctx)
weight = nd.random.normal(shape=(1024, 1024), ctx=ctx)
bias = nd.random.normal(shape=(1024,), ctx=ctx)
inputs = [{"data": data, "weight": weight, "bias": bias, "num_hidden": 1024, "no_bias": False}]

fully_connected_res = run_performance_test(
    op,
    run_backward=True,
    dtype='float32',
    ctx=ctx,
    inputs=inputs,
    warmup=10,
    runs=25
)

with open('FullyConnected.json', 'w') as f:
    json.dump(fully_connected_res, f, indent=4)

print(json.dumps(fully_connected_res, indent=4))
