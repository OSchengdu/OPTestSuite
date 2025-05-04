import mxnet as mx
from mxnet import nd
from benchmark.opperf.utils.benchmark_utils import run_performance_test
import json
ctx = mx.cpu()
op = nd.BatchNorm
data = nd.random.normal(shape=(1024, 1024, 32, 32), ctx=ctx)
gamma = nd.random.normal(shape=(1024,), ctx=ctx)
beta = nd.random.normal(shape=(1024,), ctx=ctx)
moving_mean = nd.random.normal(shape=(1024,), ctx=ctx)
moving_var = nd.random.normal(shape=(1024,), ctx=ctx)
inputs = [{"data": data, "gamma": gamma, "beta": beta, "moving_mean": moving_mean, "moving_var": moving_var, "eps": 1e-5, "momentum": 0.9}]

batch_norm_res = run_performance_test(
    op,
    run_backward=True,
    dtype='float32',
    ctx=ctx,
    inputs=inputs,
    warmup=10,
    runs=25
)

with open('BatchNorm.json', 'w') as f:
    json.dump(batch_norm_res, f, indent=4)

print(json.dumps(batch_norm_res, indent=4))
