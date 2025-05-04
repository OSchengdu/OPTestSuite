import mxnet as mx
from mxnet import nd
from benchmark.opperf.utils.benchmark_utils import run_performance_test
import json

ctx = mx.cpu() 
op = nd.cumsum

data = nd.random.normal(shape=(1024, 1024), ctx=ctx)
inputs = [{"data": data, "axis": 0}]

cumsum_res = run_performance_test(
    op,
    run_backward=True,
    dtype='float32',
    ctx=ctx,
    inputs=inputs,
    warmup=10,
    runs=25
)

with open('cumsum.json', 'w') as f:
    json.dump(cumsum_res, f, indent=4)

print(json.dumps(cumsum_res, indent=4))
