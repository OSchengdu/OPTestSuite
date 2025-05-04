import mxnet as mx
from mxnet import nd
from benchmark.opperf.utils.benchmark_utils import run_performance_test
import json

ctx = mx.cpu()
operators = [
    (nd.dot, [{"lhs": (1024, 1024), "rhs": (1024, 1024)}])
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

with open('dot.json', 'w') as f:
    json.dump(results, f, indent=4)

print(json.dumps(results, indent=4))
