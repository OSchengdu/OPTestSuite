import mxnet as mx
from mxnet import nd
from benchmark.opperf.utils.benchmark_utils import run_performance_test
import json

ctx = mx.cpu()
op = nd.linalg_extracttrian

inputs = [{"A": (1024, 1024)}]

linalg_extracttrian_res = run_performance_test(
    op,
    run_backward=True,
    dtype='float32',
    ctx=ctx,
    inputs=inputs,
    warmup=10,
    runs=25
)

with open('linalg_extracttrian.json', 'w') as f:
    json.dump(linalg_extracttrian_res, f, indent=4)

print(json.dumps(linalg_extracttrian_res, indent=4))
