import mxnet as mx
from mxnet import nd
import json

ctx = mx.cpu()

def sum_pool_1d(data, kernel, stride, pad):
    # Create a kernel that sums the values
    kernel_size = (1, kernel)
    kernel_weights = nd.ones(kernel_size, ctx=ctx) / kernel
    
    # Perform 1D convolution with the summing kernel
    conv = nd.Convolution(data, weight=kernel_weights, bias=None, kernel=kernel_size, stride=(1, stride), pad=(0, pad), num_filter=1, no_bias=True)
    
    return conv

data = nd.random.normal(shape=(1, 1, 32), ctx=ctx)
inputs = {"data": data, "kernel": 3, "stride": 1, "pad": 0}

sum_pool_1d_res = sum_pool_1d(**inputs)

with open('SumPool1D.json', 'w') as f:
    json.dump(sum_pool_1d_res.asnumpy().tolist(), f, indent=4)

print(json.dumps(sum_pool_1d_res.asnumpy().tolist(), indent=4))
