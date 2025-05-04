import mxnet as mx
from mxnet import nd
from benchmark.opperf.utils.benchmark_utils import run_performance_test
import json

ctx = mx.cpu()  # 你可以根据需要更改为 mx.gpu()

op = nd.sample_normal

inputs = [{"loc": 0.0, "scale": 1.0, "shape": (1024, 1024)}]

sample_normal_res = run_performance_test(
    op,
    run_backward=True,
    dtype='float32',
    ctx=ctx,
    inputs=inputs,
    warmup=10,
    runs=25
)

with open('mxnet_operator_benchmark_results.json', 'w') as f:
    json.dump(sample_normal_res, f, indent=4)

print(json.dumps(sample_normal_res, indent=4))
