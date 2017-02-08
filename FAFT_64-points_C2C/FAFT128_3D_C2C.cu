// Fast Accurate Fourier Transform (FAFT) was written by Oscar R. Cabrera L.
// Contributors: Renan Cabrera, Denys I. Bondar.
// Copyright (c) 2016
// All rights reserved.

#include "FAFTp128_C2C.h"

// axSplit 0

__global__ void FAFT128_C2C_ax0_dev( float2 *dataComplex, float dx, float delta, int segment, float normFactor )
{
    int tid = threadIdx.x;
    size_t sector = blockIdx.y*gridDim.x + blockIdx.x;
       
    dataComplex += ((sector>>6)<<12) + (sector & 63) + (tid<<6);

    float2 y[16];
    
    load128_half_C2C_ax0( 8, y, dataComplex, 16 );
    
    GENERAL_FAFT128( y, dx, delta, segment, tid );
    
    if ( normFactor != 0.0f ) normalize( 16, y, normFactor );
    
    store128_half_C2C_ax0<8>( y, dataComplex, 16 );
}

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

// ax Split 2

__global__ void FAFT128_C2C_ax2_dev( float2 *dataComplex, float dx, float delta, int segment, float normFactor )
{
    int tid = threadIdx.x;
    size_t sector = blockIdx.y*gridDim.x + blockIdx.x;
	
    dataComplex += ((sector>>12)<<18) + (sector & ((1<<12)-1)) + (tid<<12);
    
    float2 y[16];
    
    load128_half_C2C_ax2( 8, y, dataComplex, 16 );
			
    GENERAL_FAFT128( y, dx, delta, segment, tid );
    
    if ( normFactor != 0.0f ) normalize( 16, y, normFactor );
    
    store128_half_C2C_ax2<8>( y, dataComplex, 16 );
}

extern "C" int FAFT128_3D_C2C( float2 *dataComplex, float dx, float delta, int segment, int axes, float normFactor )
{
	int success = 1;
	dim3 grid_C2C(64*64, 1);
		
	switch(axes)
	{
		case 0:			
			FAFT128_C2C_ax0_dev<<< grid_C2C, 16 >>>( dataComplex, dx, delta, segment, normFactor );			
			break;		
			
		case 1:
			FAFT128_C2C_ax1_dev<<< grid_C2C, 16 >>>( dataComplex, dx, delta, segment, normFactor );			
			break;
			
		case 2:
			FAFT128_C2C_ax2_dev<<< grid_C2C, 16 >>>( dataComplex, dx, delta, segment, normFactor );
			break;
			
		default:
			success = 0;
			break;
	}
		
	cudaThreadSynchronize();
	
	return success;
}

