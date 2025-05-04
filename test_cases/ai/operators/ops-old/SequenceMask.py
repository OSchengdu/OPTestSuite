import mxnet as mx
from mxnet import nd
from benchmark.opperf.utils.benchmark_utils import run_performance_test
import json

ctx = mx.cpu()
op = nd.SequenceMask

data = nd.random.normal(shape=(10, 1024), ctx=ctx)
inputs = [{"data": data, "use_sequence_length": False, "sequence_length": None, "value": 0.0, "axis": 0}]

sequence_mask_res = run_performance_test(
    op,
    run_backward=True,
    dtype='float32',
    ctx=ctx,
    inputs=inputs,
    warmup=10,
    runs=25
)

with open('SequenceMask.json', 'w') as f:
    json.dump(sequence_mask_res, f, indent=4)

print(json.dumps(sequence_mask_res, indent=4))
