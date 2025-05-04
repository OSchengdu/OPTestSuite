#!/usr/bin/python
import mxnet as mx
from mxnet import nd

from benchmark.opperf.utils.benchmark_utils import run_performance_test

add_res = run_performance_test(nd.add, run_backward=True, dtype='float32', ctx=mx.cpu(),
                               inputs=[{"lhs": (1024, 1024),
                                        "rhs": (1024, 1024)}],
                               warmup=10, runs=25)
print(add_res)
