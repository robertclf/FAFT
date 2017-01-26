CUDA_INSTALL_PATH ?= /usr/local/cuda
INCLUDES  := -I. -I$(CUDA_INSTALL_PATH)/include

# You must change this parameter according to your gpu capabilities "-arch=sm_12"

NVCC      := nvcc -arch=sm_20 -Xcompiler -fPIC -shared 

all : FFT

FFT : FAFT4096_1D_C2C.so 

FAFT4096_1D_C2C.so : FAFT4096_1D_C2C.cu
	$(NVCC) $(INCLUDES) -o FAFT4096_1D_C2C.so FAFT4096_1D_C2C.cu
  
clean :
	rm -f *.so
