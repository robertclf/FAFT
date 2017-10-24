CUDA_INSTALL_PATH ?= /usr/local/cuda
INCLUDES  := -I. -I$(CUDA_INSTALL_PATH)/include
LIB       := -L$(CUDA_INSTALL_PATH)/lib64 -lcuda

# You must change this parameter according to your gpu capabilities "-arch=sm_12"

NVCC      := nvcc -arch=sm_20 -Xcompiler -fPIC -shared 

all : FFT

FFT : FAFT_64_128_4D_Z2Z.so
	
FAFT_64_128_4D_Z2Z.so : FAFT_64_128_4D_Z2Z.cu
	$(NVCC) $(LIB) $(INCLUDES) -o FAFT_64_128_4D_Z2Z.so FAFT_64_128_4D_Z2Z.cu
	
clean :
	rm -f *.so
