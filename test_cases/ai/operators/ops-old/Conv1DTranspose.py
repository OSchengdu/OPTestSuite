import mxnet as mx
from mxnet import nd
from benchmark.opperf.utils.benchmark_utils import run_performance_test
import json

ctx = mx.cpu()
op = nd.Conv1DTranspose

data = nd.random.normal(shape=(1, 3, 32), ctx=ctx)
weight = nd.random.normal(shape=(3, 16, 3), ctx=ctx)
bias = nd.random.normal(shape=(16,), ctx=ctx)
inputs = [{"data": data, "weight": weight, "bias": bias, "kernel": 3, "stride": 1, "dilate": 1, "pad": 0, "num_filter": 16, "num_group": 1, "no_bias": False, "adj": 0}]

conv_1d_transpose_res = run_performance_test(
    op,
    run_backward=True,
    dtype='float32',
    ctx=ctx,
    inputs=inputs,
    warmup=10,
    runs=25
)

with open('mxnet_operator_benchmark_results.json', 'w') as f:
    json.dump(conv_1d_transpose_res, f, indent=4)

print(json.dumps(conv_1d_transpose_res, indent=4))
