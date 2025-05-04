import mxnet as mx
from mxnet import nd
from benchmark.opperf.utils.benchmark_utils import run_performance_test
import json

ctx = mx.cpu()
op = nd.hard_sigmoid

data = nd.random.normal(shape=(1024, 1024), ctx=ctx)
inputs = [{"data": data, "alpha": 0.2, "beta": 0.5}]

hard_sigmoid_res = run_performance_test(
    op,
    run_backward=True,
    dtype='float32',
    ctx=ctx,
    inputs=inputs,
    warmup=10,
    runs=25
)

with open('hard_sigmoid.json', 'w') as f:
    json.dump(hard_sigmoid_res, f, indent=4)

print(json.dumps(hard_sigmoid_res, indent=4))
