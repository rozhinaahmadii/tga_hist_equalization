from numba import cuda
import numpy as np

@cuda.jit
def cudakernel1(array): 
    thread_position = cuda.grid(1) 
    array[thread_position] += 0.5



array = np.array([0, 1], np.float32)
print('Initial array:', array)

print('Kernel launch: cudakernel1[1, 2](array)')
cudakernel1[1, 2](array)

print('Updated array:',array)

