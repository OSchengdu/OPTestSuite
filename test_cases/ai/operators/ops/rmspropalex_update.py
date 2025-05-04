import mxnet as mx
from mxnet import nd
from benchmark.opperf.utils.benchmark_utils import run_performance_test
import json

ctx = mx.cpu() 
op = nd.rmspropalex_update

weight = nd.random.normal(shape=(1024, 1024), ctx=ctx)
grad = nd.random.normal(shape=(1024, 1024), ctx=ctx)
n = nd.zeros(shape=(1024, 1024), ctx=ctx)
g = nd.zeros(shape=(1024, 1024), ctx=ctx)
delta = nd.zeros(shape=(1024, 1024), ctx=ctx)
inputs = [{"weight": weight, "grad": grad, "n": n, "g": g, "delta": delta, "lr": 0.01, "gamma1": 0.95, "gamma2": 0.9, "epsilon": 1e-8, "wd": 0.0001}]

rmspropalex_update_res = run_performance_test(
    op,
    run_backward=True,
    dtype='float32',
    ctx=ctx,
    inputs=inputs,
    warmup=10,
    runs=25
)

with open('rmspropalex_update.json', 'w') as f:
    json.dump(rmspropalex_update_res, f, indent=4)

print(json.dumps(rmspropalex_update_res, indent=4))
