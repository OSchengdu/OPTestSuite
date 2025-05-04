import mxnet as mx
from mxnet import nd
from benchmark.opperf.utils.benchmark_utils import run_performance_test
import json

ctx = mx.cpu()  # 你可以根据需要更改为 mx.gpu()

op = nd.preloaded_multi_mp_sgd_mom_update

weights = [nd.random.normal(shape=(1024, 1024), ctx=ctx) for _ in range(5)]
grads = [nd.random.normal(shape=(1024, 1024), ctx=ctx) for _ in range(5)]
moms = [nd.zeros(shape=(1024, 1024), ctx=ctx) for _ in range(5)]
weight32s = [nd.cast(w, dtype='float32') for w in weights]
grad32s = [nd.cast(g, dtype='float32') for g in grads]
inputs = [{"weights": weights, "grads": grads, "moms": moms, "weight32s": weight32s, "grad32s": grad32s, "lr": 0.01, "momentum": 0.9, "wd": 0.0001}]

preloaded_multi_mp_sgd_mom_update_res = run_performance_test(
    op,
    run_backward=True,
    dtype='float32',
    ctx=ctx,
    inputs=inputs,
    warmup=10,
    runs=25
)

with open('mxnet_operator_benchmark_results.json', 'w') as f:
    json.dump(preloaded_multi_mp_sgd_mom_update_res, f, indent=4)

print(json.dumps(preloaded_multi_mp_sgd_mom_update_res, indent=4))
