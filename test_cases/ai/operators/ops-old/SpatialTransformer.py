import mxnet as mx
from mxnet import nd
from benchmark.opperf.utils.benchmark_utils import run_performance_test
import json

ctx = mx.cpu()
op = nd.SpatialTransformer

data = nd.random.normal(shape=(1, 3, 32, 32), ctx=ctx)
loc = nd.random.normal(shape=(1, 6, 32, 32), ctx=ctx)
inputs = [{"data": data, "loc": loc, "target_shape": (32, 32), "transform_type": "affine", "sampler_type": "bilinear"}]

spatial_transformer_res = run_performance_test(
    op,
    run_backward=True,
    dtype='float32',
    ctx=ctx,
    inputs=inputs,
    warmup=10,
    runs=25
)

with open('SpatialTransformer.json', 'w') as f:
    json.dump(spatial_transformer_res, f, indent=4)

print(json.dumps(spatial_transformer_res, indent=4))
