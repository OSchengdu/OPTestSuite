import mxnet as mx
from mxnet import nd
from benchmark.opperf.utils.benchmark_utils import run_performance_test
import json

ctx = mx.cpu() 

op = nd.smooth_l1

data = nd.random.normal(shape=(1024, 1024), ctx=ctx)
inputs = [{"data": data, "scalar": 1.0}]
smooth_l1_res = run_performance_test(
    op,
    run_backward=True,
    dtype='float32',
    ctx=ctx,
    inputs=inputs,
    warmup=10,
    runs=25
)

with open('smooth_l1.json', 'w') as f:
    json.dump(smooth_l1_res, f, indent=4)

print(json.dumps(smooth_l1_res, indent=4))
