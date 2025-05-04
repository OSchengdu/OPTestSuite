import mxnet as mx
from mxnet import nd
from benchmark.opperf.utils.benchmark_utils import run_performance_test
import json

ctx = mx.cpu()
op = nd.LRN

data = nd.random.normal(shape=(1, 3, 224, 224), ctx=ctx)
inputs = [{"data": data, "alpha": 0.0001, "beta": 0.75, "knorm": 2, "nsize": 5}]

lrn_res = run_performance_test(
    op,
    run_backward=True,
    dtype='float32',
    ctx=ctx,
    inputs=inputs,
    warmup=10,
    runs=25
)

with open('LRN.json', 'w') as f:
    json.dump(lrn_res, f, indent=4)

print(json.dumps(lrn_res, indent=4))
