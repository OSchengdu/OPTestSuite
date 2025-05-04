import mxnet as mx
from mxnet import nd
from benchmark.opperf.utils.benchmark_utils import run_performance_test
import json

ctx = mx.cpu() 
op = nd.rmsprop_update

weight = nd.random.normal(shape=(1024, 1024), ctx=ctx)
grad = nd.random.normal(shape=(1024, 1024), ctx=ctx)
n = nd.zeros(shape=(1024, 1024), ctx=ctx)
inputs = [{"weight": weight, "grad": grad, "n": n, "lr": 0.01, "gamma1": 0.9, "epsilon": 1e-8, "wd": 0.0001}]

rmsprop_update_res = run_performance_test(
    op,
    run_backward=True,
    dtype='float32',
    ctx=ctx,
    inputs=inputs,
    warmup=10,
    runs=25
)

with open('rmsporop_update.json', 'w') as f:
    json.dump(rmsprop_update_res, f, indent=4)

print(json.dumps(rmsprop_update_res, indent=4))
