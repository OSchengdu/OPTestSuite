import mxnet as mx
from mxnet import nd
from benchmark.opperf.utils.benchmark_utils import run_performance_test
import json

ctx = mx.cpu()
op = nd.unravel_index

inputs = [{"indices": (1024,), "shape": (32, 32)}]

result = run_performance_test(
    op,
    run_backward=False,
    dtype='float32',
    ctx=ctx,
    inputs=inputs,
    warmup=10,
    runs=25
)

with open('unravel_index.json', 'w') as f:
    json.dump(result, f, indent=4)

print(json.dumps(result, indent=4))
