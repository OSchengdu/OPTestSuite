import mxnet as mx
import numpy as np

# Define the context (CPU in this case)
ctx = mx.cpu()

# Helper function to print the shape of a NDArray
def print_shape(arr, name):
        print(f"{name} shape: {arr.shape}")

        # Create NDArrays for testing
        a = mx.nd.array([[1, 2], [3, 4]], ctx=ctx)
        b = mx.nd.array([[1, 1]], ctx=ctx)

        # Test broadcast_sub
        c = mx.nd.broadcast_sub(a, b)
        print_shape(c, "broadcast_sub")
        print(c)

        # Test SetValueOp
        d = mx.nd.zeros_like(a)
        d[:] = 5
        print_shape(d, "SetValueOp")
        print(d)

        # Test backward_broadcast_sub
        a.attach_grad()
        with mx.autograd.record():
                e = mx.nd.broadcast_sub(a, b)
                e.backward()
                print_shape(a.grad, "backward_broadcast_sub")
                print(a.grad)

                # Test DeleteVariable
                # MXNet automatically manages memory, so explicit deletion is not typically required
                # However, you can use the `dispose` method to manually free memory
                a.dispose()

                # Test MXNet C API Calls
                # These are low-level API calls and are not typically used directly in Python
                # They are handled by the MXNet library internally

                # Test MXNet C API Concurrency
                # This is also handled internally by MXNet and not typically exposed in Python

                # Test MXAutogradSetIsRecording
                mx.autograd.set_recording(True)
                print("MXAutogradSetIsRecording: True")

                # Test MXAutogradSetIsTraining
                mx.autograd.set_training(True)
                print("MXAutogradSetIsTraining: True")

                # Test MXImperativeInvokeEx
                # This is a low-level API call and is not typically used directly in Python

                # Test MXAutogradBackwardEx
                # This is a low-level API call and is not typically used directly in Python

                # Test MXNDArrayWaitAll
                mx.nd.waitall()
                print("MXNDArrayWaitAll: Completed")

                # Test MXNDArrayFree
                # MXNet automatically manages memory, so explicit deletion is not typically required
                # However, you can use the `dispose` method to manually free memory
                b.dispose()

                # Test Memory: cpu/0
                # MXNet automatically manages memory, so explicit deletion is not typically required
                # However, you can use the `dispose` method to manually free memory
                d.dispose()

                print("All tests completed.")
