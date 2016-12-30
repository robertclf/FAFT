// Fast Accurate Fourier Transform (FAFT) was written by Oscar R. Cabrera L.
// Contributors: Renan Cabrera, Denys I. Bondar.

#include "FAFTp_R2C_C2R.h"

// ax Split 1

__global__ void IFAFT128_C2R_ax1_dev( float *re, float *im, float2 *data65, float dx, float delta, int segment )
{
    int tid = threadIdx.x;
    size_t sector = blockIdx.y*gridDim.x + blockIdx.x;

    re += (sector*64) + tid;
    im += (sector*64) + tid + 32;
    if (tid == 0) data65 += sector;

    float2 y[16];
    
    load128_half_C2R_ax1( 8, y, re, im, data65, 16, tid );
    
    GENERAL_FAFT128( y, dx, delta, segment, tid );
       
    store128_half_C2R_ax1<8>( y, re, data65, 16, tid );
}

extern "C" int IFAFT128_1D_C2R( float *data, float2 *data65, float dx, float delta, int segment  )
{
	int success = 1;
	dim3 grid_C2R(1, 1);
	
	IFAFT128_C2R_ax1_dev<<< grid_C2R, 16 >>>( data, data, data65, dx, delta, segment  );
	
	cudaThreadSynchronize();
	
	return success;
}

