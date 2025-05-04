import mxnet as mx
from mxnet import nd
from benchmark.opperf.utils.benchmark_utils import run_performance_test
import json

ctx = mx.cpu()
op = nd.SoftmaxActivation

data = nd.random.normal(shape=(1024, 10), ctx=ctx)
inputs = [{"data": data, "mode": "channel"}]

softmax_activation_res = run_performance_test(
    op,
    run_backward=True,
    dtype='float32',
    ctx=ctx,
    inputs=inputs,
    warmup=10,
    runs=25
)

with open('SoftmaxActivation.json', 'w') as f:
    json.dump(softmax_activation_res, f, indent=4)
print(json.dumps(softmax_activation_res, indent=4))
