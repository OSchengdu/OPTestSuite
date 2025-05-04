import mxnet as mx
from mxnet import nd
from benchmark.opperf.utils.benchmark_utils import run_performance_test
import json

ctx = mx.cpu()  # 你可以根据需要更改为 mx.gpu()

op = nd.sample_poisson

inputs = [{"lam": 1.0, "shape": (1024, 1024)}]

sample_poisson_res = run_performance_test(
    op,
    run_backward=True,
    dtype='float32',
    ctx=ctx,
    inputs=inputs,
    warmup=10,
    runs=25
)

with open('mxnet_operator_benchmark_results.json', 'w') as f:
    json.dump(sample_poisson_res, f, indent=4)

print(json.dumps(sample_poisson_res, indent=4))
