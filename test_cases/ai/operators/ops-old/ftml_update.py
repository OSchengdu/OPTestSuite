import mxnet as mx
from mxnet import nd
from benchmark.opperf.utils.benchmark_utils import run_performance_test
import json

ctx = mx.cpu() 
op = nd.ftml_update

weight = nd.random.normal(shape=(1024, 1024), ctx=ctx)
grad = nd.random.normal(shape=(1024, 1024), ctx=ctx)
d = nd.zeros(shape=(1024, 1024), ctx=ctx)
v = nd.zeros(shape=(1024, 1024), ctx=ctx)
z = nd.zeros(shape=(1024, 1024), ctx=ctx)
inputs = [{"weight": weight, "grad": grad, "d": d, "v": v, "z": z, "lr": 0.01, "t": 1, "beta1": 0.6, "beta2": 0.999, "epsilon": 1e-8, "wd": 0.0001}]

ftml_update_res = run_performance_test(
    op,
    run_backward=True,
    dtype='float32',
    ctx=ctx,
    inputs=inputs,
    warmup=10,
    runs=25
)

with open('ftml_update.json', 'w') as f:
    json.dump(ftml_update_res, f, indent=4)

print(json.dumps(ftml_update_res, indent=4))
