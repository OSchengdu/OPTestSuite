import mxnet as mx
from mxnet import nd
from benchmark.opperf.utils.benchmark_utils import run_performance_test
import json

ctx = mx.cpu() 
op = nd.softmax_cross_entropy

data = nd.random.normal(shape=(1024, 10), ctx=ctx)
label = nd.random.randint(0, 10, shape=(1024,), ctx=ctx)
inputs = [{"data": data, "label": label}]

softmax_cross_entropy_res = run_performance_test(
    op,
    run_backward=True,
    dtype='float32',
    ctx=ctx,
    inputs=inputs,
    warmup=10,
    runs=25
)

with open('softmax_cross_entropy.json', 'w') as f:
    json.dump(softmax_cross_entropy_res, f, indent=4)

print(json.dumps(softmax_cross_entropy_res, indent=4))
