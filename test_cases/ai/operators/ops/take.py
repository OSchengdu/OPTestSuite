import mxnet as mx
from mxnet import nd
from benchmark.opperf.utils.benchmark_utils import run_performance_test
import json

ctx = mx.cpu()
op = nd.take

inputs = [{"a": (1024, 1024), "indices": (512,)}]

take_res = run_performance_test(
    op,
    run_backward=True,
    dtype='float32',
    ctx=ctx,
    inputs=inputs,
    warmup=10,
    runs=25
)

with open('take.json', 'w') as f:
    json.dump(take_res, f, indent=4)

print(json.dumps(take_res, indent=4))
