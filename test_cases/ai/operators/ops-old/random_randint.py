import mxnet as mx
from mxnet import nd
from benchmark.opperf.utils.benchmark_utils import run_performance_test
import json

ctx = mx.cpu()  # 你可以根据需要更改为 mx.gpu()

op = nd.random_randint

inputs = [{"shape": (1024, 1024), "low": 0, "high": 10}]

random_randint_res = run_performance_test(
    op,
    run_backward=True,
    dtype='float32',
    ctx=ctx,
    inputs=inputs,
    warmup=10,
    runs=25
)

with open('mxnet_operator_benchmark_results.json', 'w') as f:
    json.dump(random_randint_res, f, indent=4)

print(json.dumps(random_randint_res, indent=4))
