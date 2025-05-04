import mxnet as mx
from mxnet import nd
from benchmark.opperf.utils.benchmark_utils import run_performance_test
import json

ctx = mx.cpu()
op = nd.L2Normalization

data = nd.random.normal(shape=(1024, 1024), ctx=ctx)
inputs = [{"data": data, "eps": 1e-10, "mode": "channel"}]

l2_normalization_res = run_performance_test(
    op,
    run_backward=True,
    dtype='float32',
    ctx=ctx,
    inputs=inputs,
    warmup=10,
    runs=25
)

with open('L2Normalization.json', 'w') as f:
    json.dump(l2_normalization_res, f, indent=4)

print(json.dumps(l2_normalization_res, indent=4))
