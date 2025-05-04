import mxnet as mx
from mxnet import nd
from benchmark.opperf.utils.benchmark_utils import run_performance_test
import json

ctx = mx.cpu() 
op = nd.UpSampling

data = nd.random.normal(shape=(1, 3, 256, 256), ctx=ctx)
inputs = [{"data": data, "scale": 2, "sample_type": "nearest"}]

up_sampling_res = run_performance_test(
    op,
    run_backward=True,
    dtype='float32',
    ctx=ctx,
    inputs=inputs,
    warmup=10,
    runs=25
)

with open('UnSampling.json', 'w') as f:
    json.dump(up_sampling_res, f, indent=4)

print(json.dumps(up_sampling_res, indent=4))
