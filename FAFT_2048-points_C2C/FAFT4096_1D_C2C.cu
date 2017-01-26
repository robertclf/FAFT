// Fast Accurate Fourier Transform (FAFT) was written by Oscar R. Cabrera L.
// Contributors: Renan Cabrera, Denys I. Bondar.
// Copyright (c) 2016
// All rights reserved.

#include "FAFTp4096_C2C.h"

// ax Split 1

__global__ void FAFT4096_C2C_ax1_dev( float2 *data, float dx, float delta, int segment )
{
    int tid = threadIdx.x;
    size_t sector = blockIdx.y*gridDim.x + blockIdx.x;
    
    data += (sector*2048) + tid;

    float2 y[16];
    
    load4096_half_C2C_ax1( 16, y, data, 256 );
    
    GENERAL_FAFT4096( y, dx, delta, segment, tid );
        
    store4096_half_C2C_ax1<16>( y, data, 256, tid );
}

extern "C" int FAFT4096_1D_C2C( float2 *data, float dx, float delta, int segment )
{
	int success = 1;
	dim3 grid_C2C(1, 1);
	
	FAFT4096_C2C_ax1_dev<<< grid_C2C, 256 >>>( data, dx, delta, segment );
	
	cudaThreadSynchronize();
	
	return success;
}
