import mxnet as mx
from mxnet import nd
from benchmark.opperf.utils.benchmark_utils import run_performance_test
import json

ctx = mx.cpu()  # 你可以根据需要更改为 mx.gpu()

op = nd.sequence_mask

data = nd.random.normal(shape=(10, 1024), ctx=ctx)
sequence_length = nd.array([5, 4, 3, 2, 1, 5, 4, 3, 2, 1], ctx=ctx)
inputs = [{"data": data, "sequence_length": sequence_length, "use_sequence_length": True, "value": 0.0, "axis": 0}]

sequence_mask_res = run_performance_test(
    op,
    run_backward=True,
    dtype='float32',
    ctx=ctx,
    inputs=inputs,
    warmup=10,
    runs=25
)

with open('mxnet_operator_benchmark_results.json', 'w') as f:
    json.dump(sequence_mask_res, f, indent=4)

print(json.dumps(sequence_mask_res, indent=4))
