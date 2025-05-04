import mxnet as mx
from mxnet import nd
from benchmark.opperf.utils.benchmark_utils import run_performance_test
import json

ctx = mx.cpu() 
op = nd.MakeLoss

data = nd.random.normal(shape=(1024, 1024), ctx=ctx)
inputs = [{"data": data, "grad_scale": 1.0, "valid_thresh": 0, "normalization": "null"}]

make_loss_res = run_performance_test(
    op,
    run_backward=True,
    dtype='float32',
    ctx=ctx,
    inputs=inputs,
    warmup=10,
    runs=25
)

with open('MakeLoss.json', 'w') as f:
    json.dump(make_loss_res, f, indent=4)

print(json.dumps(make_loss_res, indent=4))
