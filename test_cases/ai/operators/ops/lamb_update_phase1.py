import mxnet as mx
from mxnet import nd
from benchmark.opperf.utils.benchmark_utils import run_performance_test
import json

ctx = mx.cpu()  # 你可以根据需要更改为 mx.gpu()

op = nd.lamb_update_phase1

weight = nd.random.normal(shape=(1024, 1024), ctx=ctx)
grad = nd.random.normal(shape=(1024, 1024), ctx=ctx)
mean = nd.zeros(shape=(1024, 1024), ctx=ctx)
var = nd.zeros(shape=(1024, 1024), ctx=ctx)
inputs = [{"weight": weight, "grad": grad, "mean": mean, "var": var, "lr": 0.01, "beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8, "wd": 0.0001}]

lamb_update_phase1_res = run_performance_test(
    op,
    run_backward=True,
    dtype='float32',
    ctx=ctx,
    inputs=inputs,
    warmup=10,
    runs=25
)

with open('mxnet_operator_benchmark_results.json', 'w') as f:
    json.dump(lamb_update_phase1_res, f, indent=4)

print(json.dumps(lamb_update_phase1_res, indent=4))
