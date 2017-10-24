// Fast Accurate Fourier Transform (FAFT) was written by Oscar R. Cabrera L.
// Contributors: Renan Cabrera, Denys I. Bondar.
// Copyright (c) 2016
// All rights reserved.

#include "FAFTp128_Z2Z.h"


__global__ void FAFT128_Z2Z_ax0_dev( double2 *dataComplex, double dx, double delta, int segment, double normFactor )
{
    int tid = threadIdx.x;
    size_t sector = blockIdx.y*gridDim.x + blockIdx.x;
    
    dataComplex += ((sector>>6)<<12) + (sector & 63) + (tid<<6);

    double2 y[16];

    load128_half_Z2Z_ax0( 8, y, dataComplex, 16 );
    
    GENERAL_FAFT128( y, dx, delta, segment, tid );
    
    if ( normFactor != 0.0 & normFactor != 1.0 ) normalize( 16, y, normFactor );

    store128_half_Z2Z_ax0<8>( y, dataComplex, 16 );  
}

__global__ void FAFT128_Z2Z_ax1_dev( double2 *dataComplex, double dx, double delta, int segment, double normFactor )
{
    int tid = threadIdx.x;
    size_t sector = blockIdx.y*gridDim.x + blockIdx.x;
       
    dataComplex += (sector<<6) + tid;

    double2 y[16];
    
    load128_half_Z2Z_ax1( 8, y, dataComplex, 16 );
    
    GENERAL_FAFT128( y, dx, delta, segment, tid );
    
    if ( normFactor != 0.0 & normFactor != 1.0 ) normalize( 16, y, normFactor );
    
    store128_half_Z2Z_ax1<8>( y, dataComplex, 16 );  
}

__global__ void FAFT128_Z2Z_ax2_dev( double2 *dataComplex, double dx, double delta, int segment, double normFactor )
{
    int tid = threadIdx.x;
    size_t sector = blockIdx.y*gridDim.x + blockIdx.x;
	
    dataComplex +=   ((sector>>12)<<18) + (sector & ((1<<12)-1)) + (tid<<12);
    
    double2 y[16];
    
    load128_half_Z2Z_ax2( 8, y, dataComplex, 16 );
			
    GENERAL_FAFT128( y, dx, delta, segment, tid );
    
    if ( normFactor != 0.0 & normFactor != 1.0 ) normalize( 16, y, normFactor );
    
    store128_half_Z2Z_ax2<8>( y, dataComplex, 16 );
}

__global__ void FAFT128_Z2Z_ax3_dev( double2 *dataComplex, double dx, double delta, int segment, double normFactor )
{
    int tid = threadIdx.x;
    size_t sector = blockIdx.y*gridDim.x + blockIdx.x;

    dataComplex += sector + (tid<<18);
	
    double2 y[16];
    
    load128_half_Z2Z_ax3( 8, y, dataComplex, 16 );
			
    GENERAL_FAFT128( y, dx, delta, segment, tid );
    
    if ( normFactor != 0.0 & normFactor != 1.0 ) normalize( 16, y, normFactor );
    
    store128_half_Z2Z_ax3<8>( y, dataComplex, 16 );
}

extern "C" int FAFT64_4D_Z2Z( double2 *dataComplex, double dx, double delta, int segment, int axes, double normFactor )
{
	int success = 1;
	
	dim3 grid_Z2Z(64*64, 64);
	
	switch(axes)
	{
		case 0:			
			FAFT128_Z2Z_ax0_dev<<< grid_Z2Z, 16 >>>( dataComplex, dx, delta, segment, normFactor );
			break;		
			
		case 1:
			FAFT128_Z2Z_ax1_dev<<< grid_Z2Z, 16 >>>( dataComplex, dx, delta, segment, normFactor );
			break;
			
		case 2:
			FAFT128_Z2Z_ax2_dev<<< grid_Z2Z, 16 >>>( dataComplex, dx, delta, segment, normFactor );		
			break;
		
		case 3:
			FAFT128_Z2Z_ax3_dev<<< grid_Z2Z, 16 >>>( dataComplex, dx, delta, segment, normFactor );			
			break;
			
		default:
			success = 0;
			break;
	}
		
	cudaThreadSynchronize();
	
	return success;
}

extern "C" int IFAFT64_4D_Z2Z( double2 *dataComplex, double dx, double delta, int segment, int axes, double normFactor )
{
	int success = 1;
	
	dim3 grid_Z2Z(64*64, 64);
	
	switch(axes)
	{
		case 0:			
			FAFT128_Z2Z_ax0_dev<<< grid_Z2Z, 16 >>>( dataComplex, dx, -delta, segment, normFactor );
			break;		
			
		case 1:
			FAFT128_Z2Z_ax1_dev<<< grid_Z2Z, 16 >>>( dataComplex, dx, -delta, segment, normFactor );
			break;
			
		case 2:
			FAFT128_Z2Z_ax2_dev<<< grid_Z2Z, 16 >>>( dataComplex, dx, -delta, segment, normFactor );
			break;
		
		case 3:
			FAFT128_Z2Z_ax3_dev<<< grid_Z2Z, 16 >>>( dataComplex, dx, -delta, segment, normFactor );
			break;
			
		default:
			success = 0;
			break;
	}
		
	cudaThreadSynchronize();
	
	return success;
}

