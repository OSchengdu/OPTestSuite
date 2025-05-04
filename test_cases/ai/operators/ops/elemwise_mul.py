import mxnet as mx
from mxnet import nd
from benchmark.opperf.utils.benchmark_utils import run_performance_test
import json

ctx = mx.cpu() 
op = nd.elemwise_mul

inputs = [{"lhs": (1024, 1024), "rhs": (1024, 1024)}]

elemwise_mul_res = run_performance_test(
    op,
    run_backward=True,
    dtype='float32',
    ctx=ctx,
    inputs=inputs,
    warmup=10,
    runs=25
)

with open('elemwise_mul.json', 'w') as f:
    json.dump(elemwise_mul_res, f, indent=4)

print(json.dumps(elemwise_mul_res, indent=4))
