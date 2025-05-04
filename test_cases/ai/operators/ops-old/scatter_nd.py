import mxnet as mx
from mxnet import nd
from benchmark.opperf.utils.benchmark_utils import run_performance_test
import json

ctx = mx.cpu()  # 你可以根据需要更改为 mx.gpu()

op = nd.scatter_nd

data = nd.random.normal(shape=(1024, 1024), ctx=ctx)
indices = nd.array([[1, 1], [2, 2]], ctx=ctx)
shape = (3, 3)
inputs = [{"data": data, "indices": indices, "shape": shape}]

scatter_nd_res = run_performance_test(
    op,
    run_backward=True,
    dtype='float32',
    ctx=ctx,
    inputs=inputs,
    warmup=10,
    runs=25
)

with open('mxnet_operator_benchmark_results.json', 'w') as f:
    json.dump(scatter_nd_res, f, indent=4)

print(json.dumps(scatter_nd_res, indent=4))
