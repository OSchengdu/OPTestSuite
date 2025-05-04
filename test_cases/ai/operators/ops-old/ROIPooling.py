import mxnet as mx
from mxnet import nd
from benchmark.opperf.utils.benchmark_utils import run_performance_test
import json

ctx = mx.cpu()
op = nd.ROIPooling

data = nd.random.normal(shape=(1, 3, 32, 32), ctx=ctx)
rois = nd.random.uniform(0, 32, shape=(5, 5), ctx=ctx)
inputs = [{"data": data, "rois": rois, "pooled_size": (7, 7), "spatial_scale": 1.0}]

roi_pooling_res = run_performance_test(
    op,
    run_backward=True,
    dtype='float32',
    ctx=ctx,
    inputs=inputs,
    warmup=10,
    runs=25
)

with open('ROIPooling.json', 'w') as f:
    json.dump(roi_pooling_res, f, indent=4)

print(json.dumps(roi_pooling_res, indent=4))
