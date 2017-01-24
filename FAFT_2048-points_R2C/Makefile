CUDA_INSTALL_PATH ?= /usr/local/cuda
INCLUDES  := -I. -I$(CUDA_INSTALL_PATH)/include

# You must change this parameter according to your gpu capabilities "-arch=sm_20"

NVCC      := nvcc -arch=sm_20 -Xcompiler -fPIC -shared 

all : FFT

FFT : FAFT4096_1D_R2C.so IFAFT4096_1D_C2R.so 

FAFT4096_1D_R2C.so : FAFT4096_1D_R2C.cu
	$(NVCC) $(INCLUDES) -o FAFT4096_1D_R2C.so FAFT4096_1D_R2C.cu

IFAFT4096_1D_C2R.so : IFAFT4096_1D_C2R.cu
	$(NVCC) $(INCLUDES) -o IFAFT4096_1D_C2R.so IFAFT4096_1D_C2R.cu
	
clean :
	rm -f *.so
