// Fast Accurate Fourier Transform (FAFT) was written by Oscar R. Cabrera L.
// Contributors: Renan Cabrera, Denys I. Bondar.
// Copyright (c) 2016
// All rights reserved.

#include "FAFTp64_128_Z2Z.h"

// (128,128, 64, 64)

__global__ void FAFT64_Z2Z_ax2_dev( double2 *dataComplex, double dx, double delta, int segment, double normFactor )
{
    int tid = threadIdx.x;
    size_t sector = blockIdx.y*gridDim.x + blockIdx.x;
    
    dataComplex += ((sector>>6)<<12) + (sector & 63) + (tid<<6);		// 0 -> 2

    double2 y[16];

    load128_half_Z2Z_ax2( 8, y, dataComplex, 16 );
    
    GENERAL_FAFT128( y, dx, delta, segment, tid );
    
    if ( normFactor != 0.0 & normFactor != 1.0 ) normalize( 16, y, normFactor );

    store128_half_Z2Z_ax2<8>( y, dataComplex, 16 );  
}

__global__ void FAFT64_Z2Z_ax3_dev( double2 *dataComplex, double dx, double delta, int segment, double normFactor )
{
    int tid = threadIdx.x;
    size_t sector = blockIdx.y*gridDim.x + blockIdx.x;
     
    dataComplex += (sector<<6) + tid;		// 1 -> 3

    double2 y[16];
    
    load128_half_Z2Z_ax3( 8, y, dataComplex, 16 );
    
    GENERAL_FAFT128( y, dx, delta, segment, tid );
    
    if ( normFactor != 0.0 & normFactor != 1.0 ) normalize( 16, y, normFactor );
    
    store128_half_Z2Z_ax3<8>( y, dataComplex, 16 );  
}

// ***************************************
// (64,64,128,128)

__global__ void FAFT64_Z2Z_ax1_dev( double2 *dataComplex, double dx, double delta, int segment, double normFactor )
{
    int tid = threadIdx.x;
    size_t sector = blockIdx.y*gridDim.x + blockIdx.x;
	
    dataComplex += ((sector>>14)<<20) + (sector & ((1<<14)-1)) + (tid<<14);		// 2 -> 1
    
    double2 y[16];
    
    load128_half_Z2Z_ax1( 8, y, dataComplex, 16 );
			
    GENERAL_FAFT128( y, dx, delta, segment, tid );
    
    if ( normFactor != 0.0 & normFactor != 1.0 ) normalize( 16, y, normFactor );
    
    store128_half_Z2Z_ax1<8>( y, dataComplex, 16 );
}

__global__ void FAFT64_Z2Z_ax0_dev( double2 *dataComplex, double dx, double delta, int segment, double normFactor )
{
    int tid = threadIdx.x;
    size_t sector = blockIdx.y*gridDim.x + blockIdx.x;

    dataComplex += sector + (tid<<20);		// 3 -> 0
	
    double2 y[16];
    
    load128_half_Z2Z_ax0( 8, y, dataComplex, 16 );
			
    GENERAL_FAFT128( y, dx, delta, segment, tid );
    
    if ( normFactor != 0.0 & normFactor != 1.0 ) normalize( 16, y, normFactor );
    
    store128_half_Z2Z_ax0<8>( y, dataComplex, 16 );
}

// ****************************************
// ****************************************
// ****************************************

// (64,64,128,128)

__global__ void FAFT128_Z2Z_ax2_dev( double2 *dataComplex, double dx, double delta, int segment, double normFactor )	// 0 -> 2
{
    int tid = threadIdx.x;
    size_t sector = blockIdx.y*gridDim.x + blockIdx.x;
    
    dataComplex += ((sector>>7)<<14) + (sector & 127) + (tid<<7);

    double2 y[4];
    
    load256_half_Z2Z_ax2( 4, y, dataComplex, 64 );
    
    GENERAL_FAFT256( y, dx, delta, segment, tid );
    
    if ( normFactor != 0.0 & normFactor != 1.0 ) normalize( 4, y, normFactor );
    
    store256_half_Z2Z_ax2<4>( y, dataComplex, 64 );
}

__global__ void FAFT128_Z2Z_ax3_dev( double2 *dataComplex, double dx, double delta, int segment, double normFactor )	// 1 -> 3
{
    int tid = threadIdx.x;    
    size_t sector = blockIdx.y*gridDim.x + blockIdx.x;
  
    dataComplex += (sector<<7) + tid;

    double2 y[4];
    
    load256_half_Z2Z_ax3( 4, y, dataComplex, 64 );
    
    GENERAL_FAFT256( y, dx, delta, segment, tid );
    
    if ( normFactor != 0.0 & normFactor != 1.0 ) normalize( 4, y, normFactor );
    
    store256_half_Z2Z_ax3<4>( y, dataComplex, 64 );
}

// ***************************************
// (128,128, 64, 64)

__global__ void FAFT128_Z2Z_ax1_dev( double2 *dataComplex, double dx, double delta, int segment, double normFactor )	// 2 -> 1
{
    int tid = threadIdx.x;
    size_t sector = blockIdx.y*gridDim.x + blockIdx.x;
    
    dataComplex += ((sector>>12)<<19) + (sector & ((1<<12)-1)) + (tid<<12);

    double2 y[4];
    
    load256_half_Z2Z_ax1( 4, y, dataComplex, 64 );
    
    GENERAL_FAFT256( y, dx, delta, segment, tid );
    
    if ( normFactor != 0.0 & normFactor != 1.0 ) normalize( 4, y, normFactor );
    
    store256_half_Z2Z_ax1<4>( y, dataComplex, 64 );
}

__global__ void FAFT128_Z2Z_ax0_dev( double2 *dataComplex, double dx, double delta, int segment, double normFactor )	// 3 -> 0
{
    int tid = threadIdx.x;    
    size_t sector = blockIdx.y*gridDim.x + blockIdx.x;
  
    dataComplex += sector + (tid<<19);

    double2 y[4];
    
    load256_half_Z2Z_ax0( 4, y, dataComplex, 64 );
    
    GENERAL_FAFT256( y, dx, delta, segment, tid );
    
    if ( normFactor != 0.0 & normFactor != 1.0 ) normalize( 4, y, normFactor );
    
    store256_half_Z2Z_ax0<4>( y, dataComplex, 64 );
}

// ****************************************
// ****************************************
// ****************************************

extern "C" int FAFT64_4D_Z2Z( double2 *dataComplex, double dx, double delta, int segment, int axes, double normFactor )
{
	int success = 1;

	dim3 grid_Z2Z_64_64_128_128(128*128, 64);     // (64,64,128,128)
	dim3 grid_Z2Z_128_128_64_64(64*128, 128);    // (128,128,64,64)
	
	switch(axes)
	{
		case 0:			
			FAFT64_Z2Z_ax0_dev<<< grid_Z2Z_64_64_128_128, 16 >>>( dataComplex, dx, delta, segment, normFactor );
			break;		
			
		case 1:
			FAFT64_Z2Z_ax1_dev<<< grid_Z2Z_64_64_128_128, 16 >>>( dataComplex, dx, delta, segment, normFactor );
			break;
			
		case 2:
			FAFT64_Z2Z_ax2_dev<<< grid_Z2Z_128_128_64_64, 16 >>>( dataComplex, dx, delta, segment, normFactor );
			break;
		
		case 3:
			FAFT64_Z2Z_ax3_dev<<< grid_Z2Z_128_128_64_64, 16 >>>( dataComplex, dx, delta, segment, normFactor );
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

	dim3 grid_Z2Z_64_64_128_128(128*128, 64);     // (64,64,128,128)
	dim3 grid_Z2Z_128_128_64_64(64*128, 128);    // (128,128,64,64)
	
	switch(axes)
	{
		case 0:			
			FAFT64_Z2Z_ax0_dev<<< grid_Z2Z_64_64_128_128, 16 >>>( dataComplex, dx, -delta, segment, normFactor );
			break;		
			
		case 1:
			FAFT64_Z2Z_ax1_dev<<< grid_Z2Z_64_64_128_128, 16 >>>( dataComplex, dx, -delta, segment, normFactor );
			break;
			
		case 2:
			FAFT64_Z2Z_ax2_dev<<< grid_Z2Z_128_128_64_64, 16 >>>( dataComplex, dx, -delta, segment, normFactor );
			break;
		
		case 3:
			FAFT64_Z2Z_ax3_dev<<< grid_Z2Z_128_128_64_64, 16 >>>( dataComplex, dx, -delta, segment, normFactor );
			break;
			
		default:
			success = 0;
			break;
	}
		
	cudaThreadSynchronize();
	
	return success;
}

extern "C" int FAFT128_4D_Z2Z( double2 *dataComplex, double dx, double delta, int segment, int axes, double normFactor )
{
	int success = 1;

	dim3 grid_Z2Z_64_64_128_128(128*64, 64);	// (64,64,128,128)
	dim3 grid_Z2Z_128_128_64_64(64*64, 128);	// (128,128,64,64)

	switch(axes)
	{
		case 0:		
			FAFT128_Z2Z_ax0_dev<<< grid_Z2Z_128_128_64_64, 64 >>>( dataComplex, dx, delta, segment, normFactor );
			break;

		case 1:
			FAFT128_Z2Z_ax1_dev<<< grid_Z2Z_128_128_64_64, 64 >>>( dataComplex, dx, delta, segment, normFactor );
			break;

		case 2:
			FAFT128_Z2Z_ax2_dev<<< grid_Z2Z_64_64_128_128, 64 >>>( dataComplex, dx, delta, segment, normFactor );
			break;
	
		case 3:
			FAFT128_Z2Z_ax3_dev<<< grid_Z2Z_64_64_128_128, 64 >>>( dataComplex, dx, delta, segment, normFactor );
			break;

		default:
			success = 0;
			break;
	}

	cudaThreadSynchronize();

	return success;
}

extern "C" int IFAFT128_4D_Z2Z( double2 *dataComplex, double dx, double delta, int segment, int axes, double normFactor )
{
	int success = 1;

	dim3 grid_Z2Z_64_64_128_128(128*64, 64);	// (64,64,128,128)
	dim3 grid_Z2Z_128_128_64_64(64*64, 128);	// (128,128,64,64)

	switch(axes)
	{
		case 0:		
			FAFT128_Z2Z_ax0_dev<<< grid_Z2Z_128_128_64_64, 64 >>>( dataComplex, dx, -delta, segment, normFactor );
			break;

		case 1:
			FAFT128_Z2Z_ax1_dev<<< grid_Z2Z_128_128_64_64, 64 >>>( dataComplex, dx, -delta, segment, normFactor );
			break;
			
		case 2:
			FAFT128_Z2Z_ax2_dev<<< grid_Z2Z_64_64_128_128, 64 >>>( dataComplex, dx, -delta, segment, normFactor );
			break;
		
		case 3:
			FAFT128_Z2Z_ax3_dev<<< grid_Z2Z_64_64_128_128, 64 >>>( dataComplex, dx, -delta, segment, normFactor );
			break;
			
		default:
			success = 0;
			break;
	}
		
	cudaThreadSynchronize();
	
	return success;
}

