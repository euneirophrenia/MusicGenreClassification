import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import pycuda.driver as drv
import pycuda.cumath as cumath
from pycuda.elementwise import ElementwiseKernel
from pycuda.tools import make_default_context
import numpy

#todo: use real source module to improve performance and compute the real norm of vectors


def gpuErrorEvaluate(actual, expected):
    context = make_default_context()
    device = context.get_device()
    p=gpuarray.to_gpu(numpy.array(actual))- gpuarray.to_gpu(numpy.array(expected))
    res= 1.0 - gpuarray.dot(p,p)
    context.pop()
    return res


def multgpu(a,b, t=numpy.float64):
    a = numpy.array(a).astype(t)
    b = numpy.array(b).astype(t)
    a_gpu = gparray.to_gpu(a)
    b_gpu = gparray.to_gpu(b)
    return a_gpu * b_gpu

