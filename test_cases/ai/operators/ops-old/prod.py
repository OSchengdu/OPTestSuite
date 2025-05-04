import mxnet as mx
from mxnet import nd
from benchmark.opperf.utils.benchmark_utils import run_performance_test
import json

ctx = mx.cpu()
op = nd.prod

inputs = [{"data": nd.random.normal(shape=(1024, 1024), ctx=ctx)}]
prod_res = run_performance_test(
    op,
    run_backward=True,
    dtype='float32',
    ctx=ctx,
    inputs=inputs,
    warmup=10,
    runs=25
)

with open('prod.json', 'w') as f:
    json.dump(prod_res, f, indent=4)

print(json.dumps(prod_res, indent=4))
