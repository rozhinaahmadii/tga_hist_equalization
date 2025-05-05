CUDA_HOME   = /Soft/cuda/12.2.2

NVCC        = $(CUDA_HOME)/bin/nvcc
ARCH       = -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89
NVCC_FLAGS  = -O3 -Wno-deprecated-gpu-targets -I$(CUDA_HOME)/include $(ARCH) --ptxas-options=-v -I$(CUDA_HOME)/sdk/CUDALibraries/common/inc
LD_FLAGS    = -Wno-deprecated-gpu-targets -lcudart -Xlinker -rpath,$(CUDA_HOME)/lib64 -I$(CUDA_HOME)/sdk/CUDALibraries/common/lib
##LD_FLAGS    = -Wno-deprecated-gpu-targets -lcudart -Xlinker -rpath,$(CUDA_HOME)/lib64 -I$(CUDA_HOME)/sdk/CUDALibraries/common/lib
EXE	        = Hist-CPU.exe
OBJ	        = Hist-CPU.o

default: $(EXE)

Hist-CPU.o: Hist-CPU.cu
	$(NVCC) -c -o $@ Hist-CPU.cu $(NVCC_FLAGS) -I/Soft/stb/20200430

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS)

clean:
	rm -rf *.o $(EXE)
