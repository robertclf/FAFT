// Fast Accurate Fourier Transform (FAFT) was written by Oscar R. Cabrera L.
// Contributors: Renan Cabrera, Denys I. Bondar.
// Copyright (c) 2016
// All rights reserved.

#include "FAFTp128_C2C.h"

// ax Split 1

__global__ void FAFT128_C2C_ax1_dev( float2 *dataComplex, float dx, float delta, int segment, float normFactor )
{
    int tid = threadIdx.x;
    size_t sector = blockIdx.y*gridDim.x + blockIdx.x;

    dataComplex += (sector<<6) + tid;

    float2 y[16];
    
    load128_half_C2C_ax1( 8, y, dataComplex, 16 );
    
    GENERAL_FAFT128( y, dx, delta, segment, tid );
    
    if ( normFactor != 0.0f ) normalize( 16, y, normFactor );
        
    store128_half_C2C_ax1<8>( y, dataComplex, 16 );
}

extern "C" int FAFT128_1D_C2C( float2 *dataComplex, float dx, float delta, int segment, float normFactor )
{
	int success = 1;
	dim3 grid_C2C(1, 1);
	
	FAFT128_C2C_ax1_dev<<< grid_C2C, 16 >>>( dataComplex, dx, delta, segment, normFactor );
		
	cudaThreadSynchronize();
	
	return success;
}

