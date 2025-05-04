import mxnet as mx
from mxnet import nd
from benchmark.opperf.utils.benchmark_utils import run_performance_test
import json

ctx = mx.cpu()
op = nd.min_axis

inputs = [{"data": (1024, 1024), "axis": 1}]

result = run_performance_test(
    op,
    run_backward=True,
    dtype='float32',
    ctx=ctx,
    inputs=inputs,
    warmup=10,
    runs=25
)

with open('min_axis.json', 'w') as f:
    json.dump(result, f, indent=4)

print(json.dumps(result, indent=4))
