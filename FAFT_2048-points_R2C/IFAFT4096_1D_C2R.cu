// Fast Accurate Fourier Transform (FAFT) was written by Oscar R. Cabrera L.
// Contributors: Renan Cabrera, Denys I. Bondar.
// Copyright (c) 2016
// All rights reserved.

#include "FAFTp4096_R2C_C2R.h"

// ax Split 1

__global__ void IFAFT4096_C2R_ax1_dev( float *re, float *im, float2 *data1025, float dx, float delta, int segment )
{
    int tid = threadIdx.x;
    size_t sector = blockIdx.y*gridDim.x + blockIdx.x;

    re += (sector*2048) + tid;
    im += (sector*2048) + tid + 1024;
    if (tid == 0) data1025 += sector;

    float2 y[16];
    
    load4096_half_C2R_ax1( 16, y, re, im, data1025, 256, tid );
    
    GENERAL_FAFT4096( y, dx, delta, segment, tid );
           
    store4096_half_C2R_ax1<16>( y, re, data1025, 256, tid );
}

extern "C" int IFAFT4096_1D_C2R( float *data, float2 *data1025, float dx, float delta, int segment )
{
	int success = 1;
	dim3 grid_C2R(1, 1);
	
	IFAFT4096_C2R_ax1_dev<<< grid_C2R, 256 >>>( data, data, data1025, dx, delta, segment );

	cudaThreadSynchronize();
	
	return success;
}

