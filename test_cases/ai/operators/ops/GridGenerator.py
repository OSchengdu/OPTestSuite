import mxnet as mx
from mxnet import nd
from benchmark.opperf.utils.benchmark_utils import run_performance_test
import json

ctx = mx.cpu()
op = nd.GridGenerator

inputs = [{"data": (1024, 2), "transform_type": "affine"}]

result = run_performance_test(
    op,
    run_backward=True,
    dtype='float32',
    ctx=ctx,
    inputs=inputs,
    warmup=10,
    runs=25
)

with open('GridGenerator.json', 'w') as f:
    json.dump(result, f, indent=4)

print(json.dumps(result, indent=4))
