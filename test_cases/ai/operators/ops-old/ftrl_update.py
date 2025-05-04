import mxnet as mx
from mxnet import nd
from benchmark.opperf.utils.benchmark_utils import run_performance_test
import json

ctx = mx.cpu() 
op = nd.ftrl_update

weight = nd.random.normal(shape=(1024, 1024), ctx=ctx)
grad = nd.random.normal(shape=(1024, 1024), ctx=ctx)
z = nd.zeros(shape=(1024, 1024), ctx=ctx)
n = nd.zeros(shape=(1024, 1024), ctx=ctx)
inputs = [{"weight": weight, "grad": grad, "z": z, "n": n, "lr": 0.01, "lamda1": 0.01, "beta": 1, "wd": 0.0001}]

ftrl_update_res = run_performance_test(
    op,
    run_backward=True,
    dtype='float32',
    ctx=ctx,
    inputs=inputs,
    warmup=10,
    runs=25
)

with open('ftrl_update.json', 'w') as f:
    json.dump(ftrl_update_res, f, indent=4)

print(json.dumps(ftrl_update_res, indent=4))
