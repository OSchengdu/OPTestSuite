import mxnet as mx
from mxnet import nd
from benchmark.opperf.utils.benchmark_utils import run_performance_test
import json

ctx = mx.cpu()
op = nd.CTCLoss

data = nd.random.normal(shape=(10, 16, 256), ctx=ctx)
label = nd.random.randint(0, 256, shape=(16, 10), ctx=ctx)
inputs = [{"data": data, "label": label, "data_lengths": nd.array([10]*16), "label_lengths": nd.array([10]*16)}]

ctc_loss_res = run_performance_test(
    op,
    run_backward=True,
    dtype='float32',
    ctx=ctx,
    inputs=inputs,
    warmup=10,
    runs=25
)

with open('CTCLoss.json', 'w') as f:
    json.dump(ctc_loss_res, f, indent=4)

print(json.dumps(ctc_loss_res, indent=4))
