import mxnet as mx
from mxnet import nd
from benchmark.opperf.utils.benchmark_utils import run_performance_test
import json

ctx = mx.cpu()
op = nd.LeakyReLU

data = nd.random.normal(shape=(1024, 1024), ctx=ctx)
inputs = [{"data": data, "act_type": "leaky", "slope": 0.2}]

leaky_relu_res = run_performance_test(
    op,
    run_backward=True,
    dtype='float32',
    ctx=ctx,
    inputs=inputs,
    warmup=10,
    runs=25
)

with open('LeakyReLU.json', 'w') as f:
    json.dump(leaky_relu_res, f, indent=4)

print(json.dumps(leaky_relu_res, indent=4))
