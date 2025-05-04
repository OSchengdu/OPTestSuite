import mxnet as mx
from mxnet import nd
from benchmark.opperf.utils.benchmark_utils import run_performance_test
import json

ctx = mx.cpu()
op = nd.clip

data = nd.random.normal(shape=(1024, 1024), ctx=ctx)
inputs = [{"data": data, "a_min": -1.0, "a_max": 1.0}]

clip_res = run_performance_test(
    op,
    run_backward=True,
    dtype='float32',
    ctx=ctx,
    inputs=inputs,
    warmup=10,
    runs=25
)

with open('clip.json', 'w') as f:
    json.dump(clip_res, f, indent=4)

print(json.dumps(clip_res, indent=4))
