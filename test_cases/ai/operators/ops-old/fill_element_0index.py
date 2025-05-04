import mxnet as mx
from mxnet import nd
from benchmark.opperf.utils.benchmark_utils import run_performance_test
import json

ctx = mx.cpu() 
op = nd.fill_element_0index

lhs = nd.random.normal(shape=(1024, 1024), ctx=ctx)
rhs = nd.random.normal(shape=(1024, 1024), ctx=ctx)
indices = nd.array([0], ctx=ctx)
inputs = [{"lhs": lhs, "rhs": rhs, "indices": indices}]

fill_element_0index_res = run_performance_test(
    op,
    run_backward=True,
    dtype='float32',
    ctx=ctx,
    inputs=inputs,
    warmup=10,
    runs=25
)

with open('fill_element_0index.json', 'w') as f:
    json.dump(fill_element_0index_res, f, indent=4)

print(json.dumps(fill_element_0index_res, indent=4))
