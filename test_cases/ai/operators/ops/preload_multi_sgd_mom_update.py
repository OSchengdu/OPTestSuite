import mxnet as mx
from mxnet import nd
from benchmark.opperf.utils.benchmark_utils import run_performance_test
import json

ctx = mx.cpu()
op = nd.preloaded_multi_sgd_mom_update

inputs = [
    {
        "data": (1024, 1024),
        "lr": 0.01,
        "momentum": 0.9,
        "wd": 0.0001,
        "rescale_grad": 1.0,
        "clip_gradient": None
    }
]

op_res = run_performance_test(
    op,
    run_backward=True,
    dtype='float32',
    ctx=ctx,
    inputs=inputs,
    warmup=10,
    runs=25
)

result = {
    "inputs": {"data": [1024, 1024], "lr": 0.01, "momentum": 0.9, "wd": 0.0001, "rescale_grad": 1.0, "clip_gradient": None},
    "max_storage_mem_alloc_cpu/0": op_res[0]['max_storage_mem_alloc_cpu/0'],
    "avg_time_forward_preload_multi_sgd_mom_update": op_res[0]['avg_time_forward'],
    "avg_time_backward_preload_multi_sgd_mom_update": op_res[0]['avg_time_backward']
}

with open('mxnet_operator_benchmark_results.json', 'w') as f:
    json.dump(result, f, indent=4)

print(json.dumps(result, indent=4))
