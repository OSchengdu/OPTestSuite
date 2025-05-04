import mxnet as mx
from mxnet import nd
from benchmark.opperf.utils.benchmark_utils import run_performance_test
import json

ctx = mx.cpu()
op = nd.GlobalAvgPool1D

data = nd.random.normal(shape=(1, 3, 32), ctx=ctx)
inputs = [{"data": data}]

global_avg_pool_1d_res = run_performance_test(
    op,
    run_backward=True,
    dtype='float32',
    ctx=ctx,
    inputs=inputs,
    warmup=10,
    runs=25
)

with open('GlobalAvgPool1D.json', 'w') as f:
    json.dump(global_avg_pool_1d_res, f, indent=4)

print(json.dumps(global_avg_pool_1d_res, indent=4))
