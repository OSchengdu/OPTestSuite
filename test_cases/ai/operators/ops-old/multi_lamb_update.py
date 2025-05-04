import mxnet as mx
from mxnet import nd
from benchmark.opperf.utils.benchmark_utils import run_performance_test
import json

ctx = mx.cpu()
op = nd.multi_lamb_update

inputs = [{"data": [nd.random.normal(shape=(1024, 1024), ctx=ctx) for _ in range(5)], "weights": [nd.random.normal(shape=(1024, 1024), ctx=ctx) for _ in range(5)]}]

multi_lamb_update_res = run_performance_test(
    op,
    run_backward=True,
    dtype='float32',
    ctx=ctx,
    inputs=inputs,
    warmup=10,
    runs=25
)

with open('multi_lamb_update.json', 'w') as f:
    json.dump(multi_lamb_update_res, f, indent=4)

print(json.dumps(multi_lamb_update_res, indent=4))
