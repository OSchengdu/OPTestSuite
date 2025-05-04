import mxnet as mx
from mxnet import nd
from benchmark.opperf.utils.benchmark_utils import run_performance_test
import json

ctx = mx.cpu()
op = nd.random.exponential

inputs = [{"scale": 1.0, "shape": (1024, 1024)}]
random_exponential_res = run_performance_test(
    op,
    run_backward=True,
    dtype='float32',
    ctx=ctx,
    inputs=inputs,
    warmup=10,
    runs=25
)

with open('random_exponential.json', 'w') as f:
    json.dump(random_exponential_res, f, indent=4)

print(json.dumps(random_exponential_res, indent=4))
