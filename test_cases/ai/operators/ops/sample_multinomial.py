import mxnet as mx
from mxnet import nd
from benchmark.opperf.utils.benchmark_utils import run_performance_test
import json

ctx = mx.cpu()  # 你可以根据需要更改为 mx.gpu()

op = nd.sample_multinomial

inputs = [{"data": nd.random.uniform(0, 1, shape=(1024, 1024), ctx=ctx), "shape": (1024,)}]

sample_multinomial_res = run_performance_test(
    op,
    run_backward=True,
    dtype='float32',
    ctx=ctx,
    inputs=inputs,
    warmup=10,
    runs=25
)

with open('mxnet_operator_benchmark_results.json', 'w') as f:
    json.dump(sample_multinomial_res, f, indent=4)

print(json.dumps(sample_multinomial_res, indent=4))
