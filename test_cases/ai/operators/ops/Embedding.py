import mxnet as mx
from mxnet import nd
from benchmark.opperf.utils.benchmark_utils import run_performance_test
import json

ctx = mx.cpu()
op = nd.Embedding

data = nd.array([[1, 2, 3], [4, 5, 6]], ctx=ctx)
weight = nd.random.normal(shape=(10, 128), ctx=ctx)
inputs = [{"data": data, "weight": weight, "input_dim": 10, "output_dim": 128}]

embedding_res = run_performance_test(
    op,
    run_backward=True,
    dtype='float32',
    ctx=ctx,
    inputs=inputs,
    warmup=10,
    runs=25
)

with open('Embedding.json', 'w') as f:
    json.dump(embedding_res, f, indent=4)

print(json.dumps(embedding_res, indent=4))
