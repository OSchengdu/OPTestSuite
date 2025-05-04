import mxnet as mx
from mxnet import nd
from benchmark.opperf.utils.benchmark_utils import run_performance_test
import json

ctx = mx.cpu()  # 你可以根据需要更改为 mx.gpu()

op = nd.preloaded_multi_sgd_update

weights = [nd.random.normal(shape=(1024, 1024), ctx=ctx) for _ in range(5)]
grads = [nd.random.normal(shape=(1024, 1024), ctx=ctx) for _ in range(5)]
inputs = [{"weights": weights, "grads": grads, "lr": 0.01, "wd": 0.0001}]

preloaded_multi_sgd_update_res = run_performance_test(
    op,
    run_backward=True,
    dtype='float32',
    ctx=ctx,
    inputs=inputs,
    warmup=10,
    runs=25
)

with open('mxnet_operator_benchmark_results.json', 'w') as f:
    json.dump(preloaded_multi_sgd_update_res, f, indent=4)

print(json.dumps(preloaded_multi_sgd_update_res, indent=4))
