import mxnet as mx
from mxnet import nd
from benchmark.opperf.utils.benchmark_utils import run_performance_test
import json

ctx = mx.cpu()
op = nd.RNN

data = nd.random.normal(shape=(10, 32, 128), ctx=ctx)
parameters = nd.random.normal(shape=(128*4,), ctx=ctx)
inputs = [{"data": data, "parameters": parameters, "state": None, "state_cell": None, "mode": "lstm", "state_size": 128, "num_layers": 1}]

rnn_res = run_performance_test(
    op,
    run_backward=True,
    dtype='float32',
    ctx=ctx,
    inputs=inputs,
    warmup=10,
    runs=25
)

with open('RNN.json', 'w') as f:
    json.dump(rnn_res, f, indent=4)

print(json.dumps(rnn_res, indent=4))
