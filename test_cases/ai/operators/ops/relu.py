import mxnet as mx
from mxnet import nd
from benchmark.opperf.utils.benchmark_utils import run_performance_test
import json

ctx = mx.cpu() 
operators = [
    (nd.relu, [{"data": (1024, 1024)}])
]

results = []
for op, inputs in operators:
    result = run_performance_test(
        op,
        run_backward=True,
        dtype='float32',
        ctx=ctx,
        inputs=inputs,
        warmup=10,
        runs=25
    )
    results.append(result)

with open('mxnet_operator_benchmark_results_relu.json', 'w') as f:
    json.dump(results, f, indent=4)

print(json.dumps(results, indent=4))
