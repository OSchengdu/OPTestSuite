import mxnet as mx
from mxnet import nd
from benchmark.opperf.utils.benchmark_utils import run_performance_test
import json

ctx = mx.cpu()

op = nd.mp_sgd_mom_update

weight = nd.random.normal(shape=(1024, 1024), ctx=ctx)
grad = nd.random.normal(shape=(1024, 1024), ctx=ctx)
mom = nd.zeros(shape=(1024, 1024), ctx=ctx)
weight32 = nd.cast(weight, dtype='float32')
grad32 = nd.cast(grad, dtype='float32')
inputs = [{"weight": weight, "grad": grad, "mom": mom, "weight32": weight32, "grad32": grad32, "lr": 0.01, "momentum": 0.9, "wd": 0.0001}]

mp_sgd_mom_update_res = run_performance_test(
    op,
    run_backward=True,
    dtype='float32',
    ctx=ctx,
    inputs=inputs,
    warmup=10,
    runs=25
)

with open('mp_sgd_mom_update.json', 'w') as f:
    json.dump(mp_sgd_mom_update_res, f, indent=4)

print(json.dumps(mp_sgd_mom_update_res, indent=4))
