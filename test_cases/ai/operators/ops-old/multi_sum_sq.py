import mxnet as mx
from mxnet import nd
from benchmark.opperf.utils.benchmark_utils import run_performance_test
import json

ctx = mx.cpu() 
op = nd.multi_sum_sq

inputs = [{"data": [nd.random.normal(shape=(1024, 1024), ctx=ctx) for _ in range(5)]}]
multi_sum_sq_res = run_performance_test(
    op,
    run_backward=True,
    dtype='float32',
    ctx=ctx,
    inputs=inputs,
    warmup=10,
    runs=25

)



with open('multi_sum_sq.json', 'w') as f:
    json.dump(multi_sum_sq_res, f, indent=4)

print(json.dumps(multi_sum_sq_res, indent=4))
