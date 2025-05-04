import mxnet as mx
from mxnet import nd
from benchmark.opperf.utils.benchmark_utils import run_performance_test
import json

ctx = mx.cpu()
op = nd.one_hot

inputs = [{"indices": (1024,), "depth": 10, "dtype": 'float32'}]

one_hot_res = run_performance_test(
    op,
    run_backward=True,
    dtype='float32',
    ctx=ctx,
    inputs=inputs,
    warmup=10,
    runs=25
)

with open('one_dot.json', 'w') as f:
    json.dump(one_hot_res, f, indent=4)

print(json.dumps(one_hot_res, indent=4))
