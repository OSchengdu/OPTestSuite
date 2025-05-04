import mxnet as mx
from mxnet import nd
from benchmark.opperf.utils.benchmark_utils import run_performance_test
import json

ctx = mx.cpu()
op = nd.Correlation

data1 = nd.random.normal(shape=(1, 3, 32, 32), ctx=ctx)
data2 = nd.random.normal(shape=(1, 3, 32, 32), ctx=ctx)
inputs = [{"data1": data1, "data2": data2, "kernel_size": 1, "max_displacement": 4, "stride1": 1, "stride2": 1, "pad_size": 4, "is_multiply": True}]

correlation_res = run_performance_test(
    op,
    run_backward=True,
    dtype='float32',
    ctx=ctx,
    inputs=inputs,
    warmup=10,
    runs=25
)

with open('Correlation.json', 'w') as f:
    json.dump(correlation_res, f, indent=4)
print(json.dumps(correlation_res, indent=4))
