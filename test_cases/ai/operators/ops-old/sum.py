import mxnet as mx
from mxnet import nd
from benchmark.opperf.utils.benchmark_utils import run_performance_test
import json

ctx = mx.cpu()
op = nd.sum

inputs = [{"data": nd.random.normal(shape=(1024, 1024), ctx=ctx)}]
sum_res = run_performance_test(
    op,
    run_backward=True,
    dtype='float32',
    ctx=ctx,
    inputs=inputs,
    warmup=10,
    runs=25
)

with open('sum.json', 'w') as f:
    json.dump(sum_res, f, indent=4)

print(json.dumps(sum_res, indent=4))
