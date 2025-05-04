import mxnet as mx
from mxnet import nd
from benchmark.opperf.utils.benchmark_utils import run_performance_test
import json

ctx = mx.cpu()  
op = nd.sgd_update

weight = nd.random.normal(shape=(1024, 1024), ctx=ctx)
grad = nd.random.normal(shape=(1024, 1024), ctx=ctx)
inputs = [{"weight": weight, "grad": grad, "lr": 0.01, "wd": 0.0001}]

sgd_update_res = run_performance_test(
    op,
    run_backward=True,
    dtype='float32',
    ctx=ctx,
    inputs=inputs,
    warmup=10,
    runs=25
)

with open('sgd_update.json', 'w') as f:
    json.dump(sgd_update_res, f, indent=4)

print(json.dumps(sgd_update_res, indent=4))
