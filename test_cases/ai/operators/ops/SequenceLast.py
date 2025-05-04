import mxnet as mx
from mxnet import nd
from benchmark.opperf.utils.benchmark_utils import run_performance_test
import json

ctx = mx.cpu() 
op = nd.SequenceLast

data = nd.random.normal(shape=(10, 1024), ctx=ctx)
inputs = [{"data": data, "use_sequence_length": False, "sequence_length": None}]

sequence_last_res = run_performance_test(
    op,
    run_backward=True,
    dtype='float32',
    ctx=ctx,
    inputs=inputs,
    warmup=10,
    runs=25
)

with open('SequenceLast.json', 'w') as f:
    json.dump(sequence_last_res, f, indent=4)

print(json.dumps(sequence_last_res, indent=4))
