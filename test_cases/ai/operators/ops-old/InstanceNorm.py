import mxnet as mx
from mxnet import nd
from benchmark.opperf.utils.benchmark_utils import run_performance_test
import json

ctx = mx.cpu()
op = nd.InstanceNorm

data = nd.random.normal(shape=(1024, 128, 32, 32), ctx=ctx)
gamma = nd.random.normal(shape=(128,), ctx=ctx)
beta = nd.random.normal(shape=(128,), ctx=ctx)
inputs = [{"data": data, "gamma": gamma, "beta": beta, "eps": 1e-5}]

instance_norm_res = run_performance_test(
    op,
    run_backward=True,
    dtype='float32',
    ctx=ctx,
    inputs=inputs,
    warmup=10,
    runs=25
)

with open('InstanceNorm.json', 'w') as f:
    json.dump(instance_norm_res, f, indent=4)

print(json.dumps(instance_norm_res, indent=4))
