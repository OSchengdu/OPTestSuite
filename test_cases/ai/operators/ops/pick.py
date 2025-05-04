import mxnet as mx
from mxnet import nd
from benchmark.opperf.utils.benchmark_utils import run_performance_test
import json

ctx = mx.cpu() 
op = nd.pick

inputs = [{"data": (1024, 1024), "index": (1024,), "axis": 1}]

pick_res = run_performance_test(
    op,
    run_backward=True,
    dtype='float32',
    ctx=ctx,
    inputs=inputs,
    warmup=10,
    runs=25
)

with open('pick.json', 'w') as f:
    json.dump(pick_res, f, indent=4)

print(json.dumps(pick_res, indent=4))
