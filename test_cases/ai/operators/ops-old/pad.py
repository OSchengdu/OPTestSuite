import mxnet as mx
from mxnet import nd
from benchmark.opperf.utils.benchmark_utils import run_performance_test
import json

ctx = mx.cpu() 
op = nd.pad

inputs = [{"data": (1024, 1024), "mode": 'constant', "pad_width": ((0, 0), (1, 1))}]

pad_res = run_performance_test(
    op,
    run_backward=True,
    dtype='float32',
    ctx=ctx,
    inputs=inputs,
    warmup=10,
    runs=25
)

with open('mxnet_operator_benchmark_results.json', 'w') as f:
    json.dump(pad_res, f, indent=4)

print(json.dumps(pad_res, indent=4))
