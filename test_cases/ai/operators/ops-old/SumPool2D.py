import mxnet as mx
from mxnet import nd
from benchmark.opperf.utils.benchmark_utils import run_performance_test
import json

ctx = mx.cpu() 
op = nd.SumPool2D

data = nd.random.normal(shape=(1, 3, 32, 32), ctx=ctx)
inputs = [{"data": data, "kernel": (3, 3), "stride": (1, 1), "pad": (0, 0), "pooling_convention": "valid"}]

sum_pool_2d_res = run_performance_test(
    op,
    run_backward=True,
    dtype='float32',
    ctx=ctx,
    inputs=inputs,
    warmup=10,
    runs=25
)

with open('SumPool2D.json', 'w') as f:
    json.dump(sum_pool_2d_res, f, indent=4)

print(json.dumps(sum_pool_2d_res, indent=4))
