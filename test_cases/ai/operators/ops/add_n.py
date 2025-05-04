import mxnet as mx
from mxnet import nd
from benchmark.opperf.utils.benchmark_utils import run_performance_test
import json

ctx = mx.cpu()
op = nd.add_n

inputs = [[nd.random.normal(shape=(1024, 1024), ctx=ctx) for _ in range(5)]]

add_n_res = run_performance_test(
    op,
    run_backward=True,
    dtype='float32',
    ctx=ctx,
    inputs=inputs,
    warmup=10,
    runs=25
)

with open('add_n.json', 'w') as f:
    json.dump(add_n_res, f, indent=4)

print(json.dumps(add_n_res, indent=4))
