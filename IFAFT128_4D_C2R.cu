// Fast Accurate Fourier Transform (FAFT) was written by Oscar R. Cabrera L.
// Contributors: Renan Cabrera, Denys I. Bondar.

#include "FAFTp_R2C_C2R.h"

// axSplit 0

__global__ void IFAFT128_C2R_ax0_dev( float *re, float *im, float2 *data65, float dx, float delta, int segment )
{
    int tid = threadIdx.x;
    size_t sector = blockIdx.y*gridDim.x + blockIdx.x;
    
    re += ((sector>>6)<<12) + (sector & 63) + (tid<<6);
    im += ((sector>>6)<<12) + (sector & 63) + (tid<<6) + (1<<11);
    if (tid == 0) data65 += sector;

    float2 y[16];
    
    load128_half_C2R_ax0( 8, y, re, im, data65, 16, tid );
    
    GENERAL_FAFT128( y, dx, delta, segment, tid );
        
    store128_half_C2R_ax0<8>( y, re, data65, 16, tid );
}

__global__ void IFAFT128_C2C_ax1_axSplit0_dev( float *re, float *im, float dx, float delta, int segment )
{
    int tid = threadIdx.x;
    size_t sector = blockIdx.y*gridDim.x + blockIdx.x;
    
    re += ((sector>>5)<<11) + (sector<<6) + tid;
    im += ((sector>>5)<<11) + (sector<<6) + tid  + (1<<11);
	
    float2 y[16];
    
    load128_half_C2C_ax1_axSplit0( 8, y, re, im, 16 );
    
    GENERAL_FAFT128( y, dx, delta, segment, tid );
        
    store128_half_C2C_ax1_axSplit0<8>( y, re, im, 16 );
}

__global__ void IFAFT128_C2C_65_ax1_axSplit0_dev( float2 *data65, float dx, float delta, int segment )
{
    int tid = threadIdx.x;
    size_t sector = blockIdx.y*gridDim.x + blockIdx.x;
    
    data65 += (sector<<6) + tid;	

    float2 y[16];
    
    load128_half_C2C_65_ax1_axSplit0( 8, y, data65, 16 );
    
    GENERAL_FAFT128( y, dx, delta, segment, tid );
        
    store128_half_C2C_65_ax1_axSplit0<8>( y, data65, 16 );
}

__global__ void IFAFT128_C2C_ax2_axSplit0_dev( float *re, float *im, float dx, float delta, int segment )
{
    int tid = threadIdx.x;
    size_t sector = blockIdx.y*gridDim.x + blockIdx.x;
    
    re += ((sector>>11)<<18) + (sector & ((1<<11) - 1)) + (tid<<12);
    im += ((sector>>11)<<18) + (sector & ((1<<11) - 1)) + (tid<<12) + (1<<11);
	
    float2 y[16];
    
    load128_half_C2C_ax2_axSplit0( 8, y, re, im, 16 );
    
    GENERAL_FAFT128( y, dx, delta, segment, tid );
			
    store128_half_C2C_ax2_axSplit0<8>( y, re, im, 16 );
}

__global__ void IFAFT128_C2C_65_ax2_axSplit0_dev( float2 *data65, float dx, float delta, int segment )
{
    int tid = threadIdx.x;
    size_t sector = blockIdx.y*gridDim.x + blockIdx.x;
        
    data65 += ((sector>>6)<<12) + (sector & 63) + (tid<<6);		
	
    float2 y[16];
    
    load128_half_C2C_65_ax2_axSplit0( 8, y, data65, 16 );
			
    GENERAL_FAFT128( y, dx, delta, segment, tid );

    store128_half_C2C_65_ax2_axSplit0<8>( y, data65, 16 );
}

// >>>

__global__ void IFAFT128_C2C_ax3_axSplit0_dev( float *re, float *im, float dx, float delta, int segment )
{
    int tid = threadIdx.x;
    size_t sector = blockIdx.y*gridDim.x + blockIdx.x;
           
    re += ((sector>>11)<<11) + sector + (tid<<18);
    im += ((sector>>11)<<11) + sector + (tid<<18) + (1<<11);
	
    float2 y[16];
    
    load128_half_C2C_ax3_axSplit0( 8, y, re, im, 16 );
			
    GENERAL_FAFT128( y, dx, delta, segment, tid );

    store128_half_C2C_ax3_axSplit0<8>( y, re, im, 16 );
}

__global__ void IFAFT128_C2C_65_ax3_axSplit0_dev( float2 *data65, float dx, float delta, int segment )
{
    int tid = threadIdx.x;
    size_t sector = blockIdx.y*gridDim.x + blockIdx.x;
    
    data65 += ((sector>>12)<<18) + (sector & ((1<<12)-1)) + (tid<<12);				
	
    float2 y[16];
    
    load128_half_C2C_65_ax3_axSplit0( 8, y, data65, 16 );
			
    GENERAL_FAFT128( y, dx, delta, segment, tid );

    store128_half_C2C_65_ax3_axSplit0<8>( y, data65, 16 );
}

// ***************************
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

__global__ void IFAFT128_C2C_ax0_axSplit1_dev( float *re, float *im, float dx, float delta, int segment )
{
    int tid = threadIdx.x;
    size_t sector = blockIdx.y*gridDim.x + blockIdx.x;
    
    re += ((sector>>5)<<12) + (sector & ((1<<5)-1)) + (tid<<6);
    im += ((sector>>5)<<12) + (sector & ((1<<5)-1)) + (tid<<6) + 32;
	
    float2 y[16];
    
    load128_half_C2C_ax0_axSplit1( 8, y, re, im, 16 );
    
    GENERAL_FAFT128( y, dx, delta, segment, tid );
        
    store128_half_C2C_ax0_axSplit1<8>( y, re, im, 16 );
}

__global__ void IFAFT128_C2C_65_ax0_axSplit1_dev( float2 *data65, float dx, float delta, int segment )
{
    int tid = threadIdx.x;
    size_t sector = blockIdx.y*gridDim.x + blockIdx.x;
        
    data65 += (sector<<6) + tid;		
	
    float2 y[16];
    
    load128_half_C2C_65_ax0_axSplit1( 8, y, data65, 16 );
    
    GENERAL_FAFT128( y, dx, delta, segment, tid );
        
    store128_half_C2C_65_ax0_axSplit1<8>( y, data65, 16 );
}

__global__ void IFAFT128_C2C_ax2_axSplit1_dev( float *re, float *im, float dx, float delta, int segment )
{
    int tid = threadIdx.x;
    size_t sector = blockIdx.y*gridDim.x + blockIdx.x;
    
    re += ((sector>>11)<<18) - ((sector>>11)<<11) + ((sector>>5)<<5) + (sector & ((1<<11) - 1)) + (tid<<12);		
    im += ((sector>>11)<<18) - ((sector>>11)<<11) + ((sector>>5)<<5) + (sector & ((1<<11) - 1)) + (tid<<12) + 32;
    		
    float2 y[16];
    
    load128_half_C2C_ax2_axSplit1( 8, y, re, im, 16 );
			
    GENERAL_FAFT128( y, dx, delta, segment, tid );

    store128_half_C2C_ax2_axSplit1<8>( y, re, im, 16 );
}

__global__ void IFAFT128_C2C_65_ax2_axSplit1_dev( float2 *data65, float dx, float delta, int segment )
{
    int tid = threadIdx.x;
    size_t sector = blockIdx.y*gridDim.x + blockIdx.x;
        
    data65 += ((sector>>6)<<12) + (sector & 63) + (tid<<6);		
	
    float2 y[16];
    
    load128_half_C2C_65_ax2_axSplit1( 8, y, data65, 16 );
			
    GENERAL_FAFT128( y, dx, delta, segment, tid );

    store128_half_C2C_65_ax2_axSplit1<8>( y, data65, 16 );
}

// >>>

__global__ void IFAFT128_C2C_ax3_axSplit1_dev( float *re, float *im, float dx, float delta, int segment )
{
    int tid = threadIdx.x;
    size_t sector = blockIdx.y*gridDim.x + blockIdx.x;
	    
    re += ((sector>>5)<<5) + sector + (tid<<18);   		
    im += ((sector>>5)<<5) + sector + (tid<<18) + 32;
    		
    float2 y[16];
    
    load128_half_C2C_ax3_axSplit1( 8, y, re, im, 16 );
			
    GENERAL_FAFT128( y, dx, delta, segment, tid );
    
    store128_half_C2C_ax3_axSplit1<8>( y, re, im, 16 );
}

__global__ void IFAFT128_C2C_65_ax3_axSplit1_dev( float2 *data65, float dx, float delta, int segment )
{
    int tid = threadIdx.x;
    size_t sector = blockIdx.y*gridDim.x + blockIdx.x;
    
    data65 += ((sector>>12)<<18) + (sector & ((1<<12)-1)) + (tid<<12);		
	
    float2 y[16];
    
    load128_half_C2C_65_ax3_axSplit1( 8, y, data65, 16 );
			
    GENERAL_FAFT128( y, dx, delta, segment, tid );
    
    store128_half_C2C_65_ax3_axSplit1<8>( y, data65, 16 );
}

// <<<

// ***************************
// ax Split 2

__global__ void IFAFT128_C2R_ax2_dev( float *re, float *im, float2 *data65, float dx, float delta, int segment )
{
    int tid = threadIdx.x;
    size_t sector = blockIdx.y*gridDim.x + blockIdx.x;
   
    re += ((sector>>12)<<18) + (sector & ((1<<12)-1)) + (tid<<12);
    im += ((sector>>12)<<18) + (sector & ((1<<12)-1)) + (tid<<12) + (1<<17);
    if (tid == 0) data65 += sector;
	
    float2 y[16];
    
    load128_half_C2R_ax2( 8, y, re, im, data65, 16, tid );
			
    GENERAL_FAFT128( y, dx, delta, segment, tid );
    
    store128_half_C2R_ax2<8>( y, re, data65, 16, tid );
}

__global__ void IFAFT128_C2C_ax0_axSplit2_dev( float *re, float *im, float dx, float delta, int segment )
{
    int tid = threadIdx.x;
    
    size_t sector = blockIdx.y*gridDim.x + blockIdx.x;
        		
    re += ((sector>>11)<<17) + ((sector>>6)<<12) + (sector & ((1<<6)-1)) + (tid<<6);    		
    im += ((sector>>11)<<17) + ((sector>>6)<<12) + (sector & ((1<<6)-1)) + (tid<<6) + (1<<17);

    float2 y[16];
    
    load128_half_C2C_ax0_axSplit2( 8, y, re, im, 16 );

    GENERAL_FAFT128( y, dx, delta, segment, tid );		

    store128_half_C2C_ax0_axSplit2<8>( y, re, im, 16 );
}

__global__ void IFAFT128_C2C_65_ax0_axSplit2_dev( float2 *data65, float dx, float delta, int segment )
{
    int tid = threadIdx.x;
    size_t sector = blockIdx.y*gridDim.x + blockIdx.x;
        	
    data65 += ((sector>>6)<<12) + (sector & 63) + (tid<<6);		
	
    float2 y[16];    

    load128_half_C2C_65_ax0_axSplit2( 8, y, data65, 16 );
    
    GENERAL_FAFT128( y, dx, delta, segment, tid );
    		
    store128_half_C2C_65_ax0_axSplit2<8>( y, data65, 16 );    
}

__global__ void IFAFT128_C2C_ax1_axSplit2_dev( float *re, float *im, float dx, float delta, int segment )
{
    int tid = threadIdx.x;
    size_t sector = blockIdx.y*gridDim.x + blockIdx.x;
	    
    re += ((sector>>11)<<17) + (sector<<6) + tid;
    im += ((sector>>11)<<17) + (sector<<6) + tid + (1<<17);
    		
    float2 y[16];
    
    load128_half_C2C_ax1_axSplit2( 8, y, re, im, 16 );
    
    GENERAL_FAFT128( y, dx, delta, segment, tid );
    
    store128_half_C2C_ax1_axSplit2<8>( y, re, im, 16 );
}

__global__ void IFAFT128_C2C_65_ax1_axSplit2_dev( float2 *data65, float dx, float delta, int segment )
{
    int tid = threadIdx.x;    
    size_t sector = blockIdx.y*gridDim.x + blockIdx.x;
        
    data65 += (sector<<6) + tid;		
	
    float2 y[16];
    
    load128_half_C2C_65_ax1_axSplit2( 8, y, data65, 16 );
			
    GENERAL_FAFT128( y, dx, delta, segment, tid );
    
    store128_half_C2C_65_ax1_axSplit2<8>( y, data65, 16 );
}

// >>>

__global__ void IFAFT128_C2C_ax3_axSplit2_dev( float *re, float *im, float dx, float delta, int segment )
{
    int tid = threadIdx.x;
    size_t sector = blockIdx.y*gridDim.x + blockIdx.x;
    
    re += sector + (tid<<18);
    im += sector + (tid<<18) + (1<<17);
    		
    float2 y[16];
    
    load128_half_C2C_ax3_axSplit2( 8, y, re, im, 16 );
			
    GENERAL_FAFT128( y, dx, delta, segment, tid );

    store128_half_C2C_ax3_axSplit2<8>( y, re, im, 16 );
}

__global__ void IFAFT128_C2C_65_ax3_axSplit2_dev( float2 *data65, float dx, float delta, int segment )
{
    int tid = threadIdx.x;
    size_t sector = blockIdx.y*gridDim.x + blockIdx.x;
    
    data65 += ((sector>>12)<<18) + (sector & ((1<<12)-1)) + (tid<<12);
	
    float2 y[16];
    
    load128_half_C2C_65_ax3_axSplit2( 8, y, data65, 16 );
			
    GENERAL_FAFT128( y, dx, delta, segment, tid );
    
    store128_half_C2C_65_ax3_axSplit2<8>( y, data65, 16 );
}

// <<<

// ***************************
// ax Split 3

__global__ void IFAFT128_C2R_ax3_dev( float *re, float *im, float2 *data65, float dx, float delta, int segment )
{
    int tid = threadIdx.x;
    size_t sector = blockIdx.y*gridDim.x + blockIdx.x;
   
    re += sector + (tid<<18);
    im += sector + (tid<<18) + (1<<23);
    if (tid == 0) data65 += sector;      
    
    float2 y[16];
    
    load128_half_C2R_ax3( 8, y, re, im, data65, 16, tid );
    
    GENERAL_FAFT128( y, dx, delta, segment, tid );

    store128_half_C2R_ax3<8>( y, re, data65, 16, tid );
}

__global__ void IFAFT128_C2C_ax0_axSplit3_dev( float *re, float *im, float dx, float delta, int segment )
{
    int tid = threadIdx.x;
    size_t sector = blockIdx.y*gridDim.x + blockIdx.x;
    
    re += ((sector>>6)<<12) + (sector & 63) + (tid<<6);
    im += ((sector>>6)<<12) + (sector & 63) + (tid<<6) + (1<<23);

    float2 y[16];
    
    load128_half_C2C_ax0_axSplit3( 8, y, re, im, 16 );
    
    GENERAL_FAFT128( y, dx, delta, segment, tid );
    
    store128_half_C2C_ax0_axSplit3<8>( y, re, im, 16 );    
}

__global__ void IFAFT128_C2C_65_ax0_axSplit3_dev( float2 *data65, float dx, float delta, int segment )
{
    int tid = threadIdx.x;
    size_t sector = blockIdx.y*gridDim.x + blockIdx.x;
        	
    data65 += ((sector>>6)<<12) + (sector & 63) + (tid<<6);	
	
    float2 y[16];

    load128_half_C2C_65_ax0_axSplit3( 8, y, data65, 16 );
    
    GENERAL_FAFT128( y, dx, delta, segment, tid );
			
    store128_half_C2C_65_ax0_axSplit3<8>( y, data65, 16 );
}

__global__ void IFAFT128_C2C_ax1_axSplit3_dev( float *re, float *im, float dx, float delta, int segment )
{
    int tid = threadIdx.x;
    size_t sector = blockIdx.y*gridDim.x + blockIdx.x;
    
    re += (sector<<6) + tid;
    im += (sector<<6) + tid + (1<<23);  
    		
    float2 y[16];
    
    load128_half_C2C_ax1_axSplit3( 8, y, re, im, 16 );
			
    GENERAL_FAFT128( y, dx, delta, segment, tid );
    
    store128_half_C2C_ax1_axSplit3<8>( y, re, im, 16 );
}

__global__ void IFAFT128_C2C_65_ax1_axSplit3_dev( float2 *data65, float dx, float delta, int segment )
{
    int tid = threadIdx.x;
    size_t sector = blockIdx.y*gridDim.x + blockIdx.x;
            
    data65 += (sector<<6) + tid;
	
    float2 y[16];
    
    load128_half_C2C_65_ax1_axSplit3( 8, y, data65, 16 );
			
    GENERAL_FAFT128( y, dx, delta, segment, tid );

    store128_half_C2C_65_ax1_axSplit3<8>( y, data65, 16 );
}

__global__ void IFAFT128_C2C_ax2_axSplit3_dev( float *re, float *im, float dx, float delta, int segment )
{
    int tid = threadIdx.x;
    size_t sector = blockIdx.y*gridDim.x + blockIdx.x;
    
    re += ((sector>>12)<<18) + (sector & ((1<<12)-1)) + (tid<<12);
    im += ((sector>>12)<<18) + (sector & ((1<<12)-1)) + (tid<<12) + (1<<23);
    		
    float2 y[16];
    
    load128_half_C2C_ax2_axSplit3( 8, y, re, im, 16 );
			
    GENERAL_FAFT128( y, dx, delta, segment, tid );

    store128_half_C2C_ax2_axSplit3<8>( y, re, im, 16 );
}

__global__ void IFAFT128_C2C_65_ax2_axSplit3_dev( float2 *data65, float dx, float delta, int segment )
{
    int tid = threadIdx.x;
    size_t sector = blockIdx.y*gridDim.x + blockIdx.x;
        
    data65 += ((sector>>12)<<18) + (sector & ((1<<12)-1)) + (tid<<12);
	
    float2 y[16];
    
    load128_half_C2C_65_ax2_axSplit3( 8, y, data65, 16 );
			
    GENERAL_FAFT128( y, dx, delta, segment, tid );

    store128_half_C2C_65_ax2_axSplit3<8>( y, data65, 16 );
}


extern "C" int IFAFT128_4D_C2R( float *data, float2 *data65, float beta, float delta, int segment, int axes, int makeC2R, int axesSplit )
{
	int success = 1;
	
	dim3 grid_C2R(64*64, 64);
	dim3 grid_C2C(64*64/2, 64);
	dim3 grid_C2C_65(64*64, 1);
	
	switch(axes)
	{
		case 0:
			if (makeC2R == 1){			
				IFAFT128_C2R_ax0_dev<<< grid_C2R, 16 >>>( data, data, data65, beta, delta, segment );
			}
			else{
				switch(axesSplit)
				{
					case 0:
						success = 0;
						break;
						
					case 1:
						IFAFT128_C2C_ax0_axSplit1_dev<<< grid_C2C, 16 >>>( data, data, beta, delta, segment );
				                IFAFT128_C2C_65_ax0_axSplit1_dev<<< grid_C2C_65, 16 >>>( data65, beta, delta, segment );					
						break;
						
					case 2:
						IFAFT128_C2C_ax0_axSplit2_dev<<< grid_C2C, 16 >>>( data, data, beta, delta, segment );
				                IFAFT128_C2C_65_ax0_axSplit2_dev<<< grid_C2C_65, 16 >>>( data65, beta, delta, segment );					
						break;
						
					case 3:
						IFAFT128_C2C_ax0_axSplit3_dev<<< grid_C2C, 16 >>>( data, data, beta, delta, segment );
				                IFAFT128_C2C_65_ax0_axSplit3_dev<<< grid_C2C_65, 16 >>>( data65, beta, delta, segment );					
						break;
					
					default:
						success = 0;
						break;				
				}
			}
			
			break;		
			
		case 1:
			if (makeC2R == 1){
				IFAFT128_C2R_ax1_dev<<< grid_C2R, 16 >>>( data, data, data65, beta, delta, segment  );
			}
			else{
				switch(axesSplit)
				{
					case 0:
						IFAFT128_C2C_ax1_axSplit0_dev<<< grid_C2C, 16 >>>( data, data, beta, delta, segment );
						IFAFT128_C2C_65_ax1_axSplit0_dev<<< grid_C2C_65, 16 >>>( data65, beta, delta, segment );					
						break;
					
					case 1:
						success = 0;
						break;
						
					case 2:
						IFAFT128_C2C_ax1_axSplit2_dev<<< grid_C2C, 16 >>>( data, data, beta, delta, segment );
						IFAFT128_C2C_65_ax1_axSplit2_dev<<< grid_C2C_65, 16 >>>( data65, beta, delta, segment );					
						break;
						
					case 3:
						IFAFT128_C2C_ax1_axSplit3_dev<<< grid_C2C, 16 >>>( data, data, beta, delta, segment );
						IFAFT128_C2C_65_ax1_axSplit3_dev<<< grid_C2C_65, 16 >>>( data65, beta, delta, segment );					
						break;
				
					default:
						success = 0;
						break;		
				}
			}
			
			break;

		case 2:
			if (makeC2R == 1){
				IFAFT128_C2R_ax2_dev<<< grid_C2R, 16 >>>( data, data, data65, beta, delta, segment );
			}
			else{
				switch(axesSplit)
				{
					case 0:
						IFAFT128_C2C_ax2_axSplit0_dev<<< grid_C2C, 16 >>>( data, data, beta, delta, segment );
						IFAFT128_C2C_65_ax2_axSplit0_dev<<< grid_C2C_65, 16 >>>( data65, beta, delta, segment );					
						break;
						
					case 1:
						IFAFT128_C2C_ax2_axSplit1_dev<<< grid_C2C, 16 >>>( data, data, beta, delta, segment );
						IFAFT128_C2C_65_ax2_axSplit1_dev<<< grid_C2C_65, 16 >>>( data65, beta, delta, segment );					
						break;
						
					case 2:
						success = 0;
						break;
						
					case 3:
						IFAFT128_C2C_ax2_axSplit3_dev<<< grid_C2C, 16 >>>( data, data, beta, delta, segment );
						IFAFT128_C2C_65_ax2_axSplit3_dev<<< grid_C2C_65, 16 >>>( data65, beta, delta, segment );					
						break;
				
					default:
						success = 0;
						break;		
				}
			}
			
			break;
			
		case 3:
			if (makeC2R == 1){
				IFAFT128_C2R_ax3_dev<<< grid_C2R, 16 >>>( data, data, data65, beta, delta, segment );
			}
			else{
				switch(axesSplit)
				{
					case 0:
						IFAFT128_C2C_ax3_axSplit0_dev<<< grid_C2C, 16 >>>( data, data, beta, delta, segment );
						IFAFT128_C2C_65_ax3_axSplit0_dev<<< grid_C2C_65, 16 >>>( data65, beta, delta, segment );					
						break;
						
					case 1:
						IFAFT128_C2C_ax3_axSplit1_dev<<< grid_C2C, 16 >>>( data, data, beta, delta, segment );
						IFAFT128_C2C_65_ax3_axSplit1_dev<<< grid_C2C_65, 16 >>>( data65, beta, delta, segment );					
						break;
						
					case 2:
						IFAFT128_C2C_ax3_axSplit2_dev<<< grid_C2C, 16 >>>( data, data, beta, delta, segment );
						IFAFT128_C2C_65_ax3_axSplit2_dev<<< grid_C2C_65, 16 >>>( data65, beta, delta, segment );
						break;
						
					case 3:
						success = 0;
						break;
				
					default:
						success = 0;
						break;		
				}
			}
			
			break;

		default:
			success = 0;
			break;
	}
		
	cudaThreadSynchronize();
	
	return success;
}

