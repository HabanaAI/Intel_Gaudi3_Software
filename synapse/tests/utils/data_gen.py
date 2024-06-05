
# This script generates output data for spatial conv unit test.
import mxnet as mx
import numpy as np

batch = 2
output_width = 4
output_height = 4
input_channel = 2
kernel_size = (3,3)
output_channel = 2
stride = (1,1)
pad = (0,0)

#o = ((i - k + 2 * pad) / stride) + 1

input_height = ((output_height - 1) * stride[0]) + kernel_size[0] - (2 * pad[0]);
input_width = ((output_width - 1) * stride[1]) + kernel_size[1] - (2 * pad[1]);

print ("input_height: "+ str(input_height))
print ("input_width: " +str(input_width))

ifm    = range(1,batch*input_width*input_height*input_channel+1) #NHWC
weight = range(1,kernel_size[0]*kernel_size[1]*input_channel*output_channel+1)  #RSCK

ifm = [ x % 128 for x in ifm]
weight = [ x % 128 for x in weight]

ifm = np.reshape(ifm,(batch,input_height,input_width,input_channel))
weight = np.reshape(weight,(kernel_size[0],kernel_size[1],input_channel,output_channel))



ref_ifm    = mx.nd.array(ifm,    ctx=mx.cpu())
ref_weight = mx.nd.array(weight, ctx=mx.cpu())
ref_ifm    = mx.ndarray.transpose(ref_ifm,    axes=(0,3,1,2)) #from NCHW to NHWC
ref_weight = mx.ndarray.transpose(ref_weight, axes=(3,2,0,1)) #from KCRS to RSCK

print ref_ifm.asnumpy()
print ref_weight.asnumpy()

ref_result = mx.ndarray.Convolution (ref_ifm, ref_weight, num_filter=output_channel, pad=pad, stride= stride, layout = 'NCHW', kernel = kernel_size, no_bias=True)

# transpose back to NCHW
transposed = mx.ndarray.transpose(ref_result, axes=(0,2,3,1))

transposed = mx.ndarray.Flatten(transposed)
print transposed.asnumpy()


