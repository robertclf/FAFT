CUDA_INSTALL_PATH ?= /usr/local/cuda
INCLUDES  := -I. -I$(CUDA_INSTALL_PATH)/include
#LIB       := -L$(CUDA_INSTALL_PATH)/lib64 -lcuda -lcudart

# You must change this parameter according to your gpu capabilities "-arch=sm_12"

NVCC      := nvcc -arch=sm_12 -Xcompiler -fPIC -shared 

all : FFT

FFT : FAFT128_1D_C2C.so FAFT128_2D_C2C.so FAFT128_3D_C2C.so FAFT128_4D_C2C.so
		

# C2C

FAFT128_1D_C2C.so : FAFT128_1D_C2C.cu
	$(NVCC) $(INCLUDES) -o FAFT128_1D_C2C.so FAFT128_1D_C2C.cu

FAFT128_2D_C2C.so : FAFT128_2D_C2C.cu
	$(NVCC) $(INCLUDES) -o FAFT128_2D_C2C.so FAFT128_2D_C2C.cu
	
FAFT128_3D_C2C.so : FAFT128_3D_C2C.cu
	$(NVCC) $(INCLUDES) -o FAFT128_3D_C2C.so FAFT128_3D_C2C.cu
	
FAFT128_4D_C2C.so : FAFT128_4D_C2C.cu
	$(NVCC) $(INCLUDES) -o FAFT128_4D_C2C.so FAFT128_4D_C2C.cu
	
clean :
	rm -f *.so
