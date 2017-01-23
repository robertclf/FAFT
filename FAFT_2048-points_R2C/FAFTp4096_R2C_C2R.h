// Fast Accurate Fourier Transform (FAFT) was written by Oscar R. Cabrera L.
// Contributors: Renan Cabrera, Denys I. Bondar.
// Copyright (c) 2016
// All rights reserved.
	
#pragma once

#define _USE_MATH_DEFINES
#include <math.h>

inline __device__ float2 operator_mul_zz( float2 a, float2 b ){ return make_float2( a.x*b.x-a.y*b.y, a.x*b.y+a.y*b.x ); }
inline __device__ float2 operator_mul_za( float2 a, float b  ){ return make_float2( b*a.x, b*a.y ); }
inline __device__ float2 operator_div_za( float2 a, float b  ){ return make_float2( a.x/b, a.y/b ); }
inline __device__ float2 operator_plu_zz( float2 a, float2 b ){ return make_float2( a.x + b.x, a.y + b.y ); }
inline __device__ float2 operator_min_zz( float2 a, float2 b ){ return make_float2( a.x - b.x, a.y - b.y ); }

#define cos_pi_8  0.923879533f
#define sin_pi_8  0.382683432f
#define sqrt_5_4  0.559016994f
#define sin_2pi_5 0.9510565163f
#define sin_pi_5  0.5877852523f

#define imag      make_float2( 0.0f,  1.0f )
#define imag_neg  make_float2( 0.0f, -1.0f )

#define exp_1_16  make_float2(  cos_pi_8, -sin_pi_8 )
#define exp_3_16  make_float2(  sin_pi_8, -cos_pi_8 )
#define exp_5_16  make_float2( -sin_pi_8, -cos_pi_8 )
#define exp_7_16  make_float2( -cos_pi_8, -sin_pi_8 )
#define exp_9_16  make_float2( -cos_pi_8,  sin_pi_8 )
#define exp_1_8   make_float2(  1.0f, -1.0f )
#define exp_1_4   make_float2(  0.0f, -1.0f )
#define exp_3_8   make_float2( -1.0f, -1.0f )

#define iexp_1_16  make_float2(  cos_pi_8,  sin_pi_8 )
#define iexp_3_16  make_float2(  sin_pi_8,  cos_pi_8 )
#define iexp_5_16  make_float2( -sin_pi_8,  cos_pi_8 )
#define iexp_7_16  make_float2( -cos_pi_8,  sin_pi_8 )
#define iexp_9_16  make_float2( -cos_pi_8, -sin_pi_8 )
#define iexp_1_8   make_float2(  1.0f, 1.0f )
#define iexp_1_4   make_float2(  0.0f, 1.0f )
#define iexp_3_8   make_float2( -1.0f, 1.0f )

#define M_points 2048
#define M_base 256

inline __device__ float2 exp_i( float phi )
{
    return make_float2( (float)cos(phi), (float)sin(phi) );
}

template<int radix> inline __device__ int rev( int bit );

template<> inline __device__ int rev<2>( int bit )
{
    return bit;
}

template<> inline __device__ int rev<4>( int bit )
{
    int arrev[] = {0,2,1,3};
    return arrev[bit];
}

template<> inline __device__ int rev<5>( int bit )
{
    int arrev[] = {0,4,3,2,1};
    return arrev[bit];
}

template<> inline __device__ int rev<8>( int bit )
{
    int arrev[] = {0,4,2,6,1,5,3,7};
    return arrev[bit];
}

template<> inline __device__ int rev<16>( int bit )
{
    int arrev[] = {0,8,4,12,2,10,6,14,1,9,5,13,3,11,7,15};
    return arrev[bit];
}


#define IFFT2 FFT2
inline __device__ void FFT2( float2 *z0, float2 *z1 )
{ 
    float2 t0 = *z0;
    
    *z0 = operator_plu_zz( t0, *z1 ); 
    *z1 = operator_min_zz( t0, *z1 );
}

inline __device__ void FFT4( float2 *z0, float2 *z1, float2 *z2, float2 *z3 )
{

    FFT2( z0, z2 );
    FFT2( z1, z3 );
    *z3 = operator_mul_zz( *z3, exp_1_4 );
    FFT2( z0, z1 );
    FFT2( z2, z3 );
    
}

inline __device__ void IFFT4( float2 *z0, float2 *z1, float2 *z2, float2 *z3 )
{
    IFFT2( z0, z2 );
    IFFT2( z1, z3 );
    *z3 = operator_mul_zz( *z3, iexp_1_4 );
    IFFT2( z0, z1 );
    IFFT2( z2, z3 );
}

inline __device__ void  FFT2vec( float2 *z ) {  FFT2( &z[0], &z[1] ); }
inline __device__ void IFFT2vec( float2 *z ) { IFFT2( &z[0], &z[1] ); }
inline __device__ void  FFT4vec( float2 *z ) {  FFT4( &z[0], &z[1], &z[2], &z[3] ); }
inline __device__ void IFFT4vec( float2 *z ) { IFFT4( &z[0], &z[1], &z[2], &z[3] ); }

inline __device__ void FFT8( float2 *z )
{
    FFT2( &z[0], &z[4] );
    FFT2( &z[1], &z[5] );
    FFT2( &z[2], &z[6] );
    FFT2( &z[3], &z[7] );
    
    z[5] = operator_mul_za( operator_mul_zz( z[5], exp_1_8 ), M_SQRT1_2 );
    z[6] = operator_mul_zz( z[6], exp_1_4 );
    z[7] = operator_mul_za( operator_mul_zz( z[7], exp_3_8 ), M_SQRT1_2 );

    FFT4( &z[0], &z[1], &z[2], &z[3] );
    FFT4( &z[4], &z[5], &z[6], &z[7] );
}

inline __device__ void IFFT8( float2 *z )
{
    IFFT2( &z[0], &z[4] );
    IFFT2( &z[1], &z[5] );
    IFFT2( &z[2], &z[6] );
    IFFT2( &z[3], &z[7] );
    
    z[5] = operator_mul_za( operator_mul_zz( z[5], iexp_1_8 ), M_SQRT1_2 );
    z[6] = operator_mul_zz( z[6], iexp_1_4 );
    z[7] = operator_mul_za( operator_mul_zz( z[7], iexp_3_8 ), M_SQRT1_2 );

    IFFT4( &z[0], &z[1], &z[2], &z[3] );
    IFFT4( &z[4], &z[5], &z[6], &z[7] );
}

inline __device__  void FFT5( float2 *z)
{
    float2 t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11;
    
    t1 = operator_plu_zz( z[1], z[4] ); 
    t2 = operator_plu_zz( z[2], z[3] ); 
    t3 = operator_min_zz( z[1], z[4] ); 
    t4 = operator_min_zz( z[2], z[3] );    
    
    t5 = operator_plu_zz( t1, t2 ); 
    t6 = operator_mul_za( operator_min_zz( t1, t2 ), sqrt_5_4 ); 
    t7 = operator_min_zz( z[0], operator_div_za( t5, 4 ));    
    
    t8 = operator_plu_zz( t7, t6 ); 
    t9 = operator_min_zz( t7, t6 );
    
    t10 = operator_plu_zz( operator_mul_za( t3, sin_2pi_5 ), operator_mul_za( t4, sin_pi_5  )); 
    t11 = operator_min_zz( operator_mul_za( t3, sin_pi_5  ), operator_mul_za( t4, sin_2pi_5 ));    
    
    z[0] = operator_plu_zz( z[0], t5 );
    z[1] = operator_plu_zz( t8, operator_mul_zz( imag, t10 )); 
    z[2] = operator_plu_zz( t9, operator_mul_zz( imag, t11 ));
    z[3] = operator_plu_zz( t9, operator_mul_zz( imag_neg, t11 )); 
    z[4] = operator_plu_zz( t8, operator_mul_zz( imag_neg, t10 ));
}

inline __device__  void IFFT5( float2 *z)
{
    float2 t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11;
    
    t1 = operator_plu_zz( z[1], z[4] ); 
    t2 = operator_plu_zz( z[2], z[3] ); 
    t3 = operator_min_zz( z[1], z[4] ); 
    t4 = operator_min_zz( z[2], z[3] );    
    
    t5 = operator_plu_zz( t1, t2 ); 
    t6 = operator_mul_za( operator_min_zz( t1, t2 ), sqrt_5_4 ); 
    t7 = operator_min_zz( z[0], operator_div_za( t5, 4 ));    
    
    t8 = operator_plu_zz( t7, t6 ); 
    t9 = operator_min_zz( t7, t6 );
    
    t10 = operator_plu_zz( operator_mul_za( t3, sin_2pi_5 ), operator_mul_za( t4, sin_pi_5  )); 
    t11 = operator_min_zz( operator_mul_za( t3, sin_pi_5  ), operator_mul_za( t4, sin_2pi_5 ));    
    
    z[0] = operator_plu_zz( z[0], t5 );
    z[1] = operator_plu_zz( t8, operator_mul_zz( imag_neg, t10 )); 
    z[2] = operator_plu_zz( t9, operator_mul_zz( imag_neg, t11 ));
    z[3] = operator_plu_zz( t9, operator_mul_zz( imag, t11 )); 
    z[4] = operator_plu_zz( t8, operator_mul_zz( imag, t10 ));    
}

inline __device__ void FFT16( float2 *z )
{
    FFT4( &z[0], &z[4], &z[8],  &z[12] );
    FFT4( &z[1], &z[5], &z[9],  &z[13] );
    FFT4( &z[2], &z[6], &z[10], &z[14] );
    FFT4( &z[3], &z[7], &z[11], &z[15] );

    z[5]  = operator_mul_za( operator_mul_zz( z[5], exp_1_8 ), M_SQRT1_2 );
    z[6]  = operator_mul_zz( z[6], exp_1_4 );
    z[7]  = operator_mul_za( operator_mul_zz( z[7], exp_3_8 ), M_SQRT1_2 );
    z[9]  = operator_mul_zz( z[9], exp_1_16 );
    z[10] = operator_mul_za( operator_mul_zz( z[10], exp_1_8 ), M_SQRT1_2 );
    z[11] = operator_mul_zz( z[11], exp_3_16 );
    z[13] = operator_mul_zz( z[13], exp_3_16 );
    z[14] = operator_mul_za( operator_mul_zz( z[14], exp_3_8 ), M_SQRT1_2 );
    z[15] = operator_mul_zz( z[15], exp_9_16 );

    FFT4( &z[0],  &z[1],  &z[2],  &z[3] );
    FFT4( &z[4],  &z[5],  &z[6],  &z[7] );
    FFT4( &z[8],  &z[9],  &z[10], &z[11] );
    FFT4( &z[12], &z[13], &z[14], &z[15] );
}

inline __device__ void IFFT16( float2 *z )
{
    IFFT4( &z[0], &z[4], &z[8],  &z[12] );
    IFFT4( &z[1], &z[5], &z[9],  &z[13] );
    IFFT4( &z[2], &z[6], &z[10], &z[14] );
    IFFT4( &z[3], &z[7], &z[11], &z[15] );

    z[5]  = operator_mul_za( operator_mul_zz( z[5], iexp_1_8 ), M_SQRT1_2 );
    z[6]  = operator_mul_zz( z[6], iexp_1_4 );
    z[7]  = operator_mul_za( operator_mul_zz( z[7], iexp_3_8 ), M_SQRT1_2 );
    z[9]  = operator_mul_zz( z[9], iexp_1_16 );
    z[10] = operator_mul_za( operator_mul_zz( z[10], iexp_1_8 ), M_SQRT1_2 );
    z[11] = operator_mul_zz( z[11], iexp_3_16 );
    z[13] = operator_mul_zz( z[13], iexp_3_16 );
    z[14] = operator_mul_za( operator_mul_zz( z[14], iexp_3_8 ), M_SQRT1_2 );
    z[15] = operator_mul_zz( z[15], iexp_9_16 );

    IFFT4( &z[0],  &z[1],  &z[2],  &z[3] );
    IFFT4( &z[4],  &z[5],  &z[6],  &z[7] );
    IFFT4( &z[8],  &z[9],  &z[10], &z[11] );
    IFFT4( &z[12], &z[13], &z[14], &z[15] );
}

// Multiply by Minus ONE
/*
inline __device__ void mulByMinusOne( int radix, float2 *z, size_t sector, int tid )
{
    for ( int i = 0; i < radix; i++ )
        z[i] = operator_mul_za( z[i], pow( (float)(-1), (float)( ( (sector/160) + (sector % 160) + tid ) & 1) ));   
}
*/

inline __device__ void normalize( int radix, float2 *z, float normFactor )
{
    for ( int i = 0; i < radix; i++ )
        z[i] = operator_div_za( z[i], normFactor );
}

//////////////////
//   TWIDDLES   //
//////////////////

template<int radix> inline __device__ void twiddle( float2 *z, int i, int n )
{
    for ( int j = 1; j < radix; j++ )
        z[j] = operator_mul_zz( z[j], exp_i((-2*M_PI*rev<radix>( j )/n)*i) );
}

template<int radix> inline __device__ void itwiddle( float2 *z, int i, int n )
{
    for ( int j = 1; j < radix; j++ )
        z[j] = operator_mul_zz( z[j], exp_i(( 2*M_PI*rev<radix>( j )/n)*i) );
}

///////////////////////
//   SHARED MEMORY   //
///////////////////////

inline __device__ void loadSharedx( int radix, float2 *z, float *a, int sx )
{
    for( int i = 0; i < radix; i++ )
        z[i].x = a[i*sx];
}

inline __device__ void loadSharedy( int radix, float2 *z, float *a, int sx )
{
    for( int i = 0; i < radix; i++ )
        z[i].y = a[i*sx];
}

template<int radix> inline __device__ void storeSharedx( float2 *z, float *a, int sx )
{
    #pragma unroll
    for( int i = 0; i < radix; i++ )
        a[i*sx] = z[rev<radix>( i )].x;
}

template<int radix> inline __device__ void storeSharedy( float2 *z, float *a, int sx )
{
    #pragma unroll
    for( int i = 0; i < radix; i++ )
        a[i*sx] = z[rev<radix>( i )].y;
}

/////////////////
//   FORWARD   //
/////////////////

// LOADS R2C

inline __device__ void load256_half_R2C_ax0( int radix, float2 *z, float *a, int sx )
{
    for( int i = 0; i < radix; i++ )
        if ( i < (radix>>1) )
        {
            z[i].x = a[(i*sx)<<7];
            z[i].y = 0.0f;
        }
        else
    	    z[i] = make_float2( 0.0f, 0.0f );
}

inline __device__ void load4096_half_R2C_ax1( int radix, float2 *z, float *a, int sx )
{
    for( int i = 0; i < radix; i++ )
    	if ( i < (radix>>1) )
    	{
    	    z[i].x = a[i*sx];
    	    z[i].y = 0.0f;
    	}
    	else
    	    z[i] = make_float2( 0.0f, 0.0f );       
}

inline __device__ void load4096_R2C_ax1( int radix, float2 *z, float2 *a, int sx )
{
    for( int i = 0; i < radix; i++ )
    	{
            z[i] = a[i*sx];   
    	}
}

inline __device__ void load256_half_R2C_ax2( int radix, float2 *z, float *a, int sx )
{
    for( int i = 0; i < radix; i++ )
        if ( i < (radix>>1) )
        {
            z[i].x = a[(i*sx)<<14];
            z[i].y = 0.0f;
        }
        else
            z[i] = make_float2( 0.0f, 0.0f );  
}

inline __device__ void load256_half_R2C_ax3( int radix, float2 *z, float *a, int sx )
{
    for( int i = 0; i < radix; i++ )
        if ( i < (radix>>1) )
        {
            z[i].x = a[(i*sx)<<21];
            z[i].y = 0.0f;
        }
        else
            z[i] = make_float2( 0.0f, 0.0f );
}

// STORES R2C

template<int radix> inline __device__ void store256_half_R2C_ax0( float2 *z, float *re, float *im, float2 *a129, int sx, int tid )
{
    #pragma unroll
    for( int i = 0; i < (radix>>2); i++ )
    {	
        re[(i*sx)<<7] = z[rev<radix>(i)].x;
        im[(i*sx)<<7] = z[rev<radix>(i)].y;
    }
        
    if (tid == 0) a129[0] = z[rev<radix>(radix>>2)];
}

template<int radix> inline __device__ void store4096_half_R2C_ax1( float2 *z, float *re, float *im, float2 *a129, int sx, int tid )
{
    #pragma unroll
    for( int i = 0; i < (radix>>2); i++ )  
    {
        re[(i*sx)] = z[rev<radix>(i)].x;
        im[(i*sx)] = z[rev<radix>(i)].y;
    }
    
    if (tid == 0) a129[0] = z[rev<radix>(radix>>2)];
}
/*
template<int radix> inline __device__ void store4096_R2C_ax1( float2 *z, float2 *re, float2 *im, float2 *a129, int sx, int tid )
{
    #pragma unroll
    for( int i = 0; i < radix; i++ )    
    {
        re[(i*sx)] = z[rev<radix>(i)];
        //re[(i*sx)] = z[i];
    }
}
*/
template<int radix> inline __device__ void store256_half_R2C_ax2( float2 *z, float *re, float *im, float2 *a129, int sx, int tid )
{
    #pragma unroll
    for( int i = 0; i < (radix>>2); i++ )
    {	
        re[(i*sx)<<14] = z[rev<radix>(i)].x;
        im[(i*sx)<<14] = z[rev<radix>(i)].y;
    }
        
    if (tid == 0) a129[0] = z[rev<radix>(radix>>2)];
}

template<int radix> inline __device__ void store256_half_R2C_ax3( float2 *z, float *re, float *im, float2 *a129, int sx, int tid )
{
    #pragma unroll
    for( int i = 0; i < (radix>>2); i++ )
    {	
        re[(i*sx)<<21] = z[rev<radix>(i)].x;
        im[(i*sx)<<21] = z[rev<radix>(i)].y;
    }
        
    if (tid == 0) a129[0] = z[rev<radix>(radix>>2)];
}
/////////////////
//   INVERSE   //
/////////////////

// LOADS C2R

inline __device__ void load256_half_C2R_ax0( int radix, float2 *z, float *re, float *im, float2 *data129, int sx, int tid )
{
    for( int i = 0; i < radix>>1; i++ )
    	i < (radix>>2)? z[i] = make_float2( re[(i*sx)<<7], im[(i*sx)<<7] ) : z[i] = make_float2( re[((128 - i*sx - 2*tid)<<7)], -im[((128 - i*sx - 2*tid)<<7)] );
     
    if (tid == 0) z[radix>>2] = data129[0]; 
    
    for( int i = radix>>1; i < radix; i++ )
        z[i] = make_float2( 0.0f, 0.0f );
}

inline __device__ void load4096_half_C2R_ax1( int radix, float2 *z, float *re, float *im, float2 *data1025, int sx, int tid )
{
    for( int i = 0; i < radix>>1; i++ )
    	i < radix>>2? z[i] = make_float2( re[(i*sx)], im[(i*sx)] ) : z[i] = make_float2( re[(M_points - i*sx - 2*tid)], -im[(M_points - i*sx - 2*tid)] );
     
    if (tid == 0) z[radix>>2] = data1025[0];
    
    for( int i = radix>>1; i < radix; i++ )
        z[i] = make_float2( 0.0f, 0.0f );     
}

inline __device__ void load256_half_C2R_ax2( int radix, float2 *z, float *re, float *im, float2 *data129, int sx, int tid )
{
    for( int i = 0; i < radix>>1; i++ )
    	i < radix>>2? z[i] = make_float2( re[(i*sx)<<14], im[(i*sx)<<14] ) : z[i] = make_float2( re[((128 - i*sx - 2*tid)<<14)], -im[((128 - i*sx - 2*tid)<<14)] );
     
    if (tid == 0) z[radix>>2] = data129[0]; 
    
    for( int i = radix>>1; i < radix; i++ )
        z[i] = make_float2( 0.0f, 0.0f );
}

inline __device__ void load256_half_C2R_ax3( int radix, float2 *z, float *re, float *im, float2 *data129, int sx, int tid )
{
    for( int i = 0; i < radix>>1; i++ )
    	i < radix>>2? z[i] = make_float2( re[(i*sx)<<21], im[(i*sx)<<21] ) : z[i] = make_float2( re[((128 - i*sx - 2*tid)<<21)], -im[((128 - i*sx - 2*tid)<<21)] );
     
    if (tid == 0) z[radix>>2] = data129[0]; 
    
    for( int i = radix>>1; i < radix; i++ )
        z[i] = make_float2( 0.0f, 0.0f );
}

// STORES C2R

template<int radix> inline __device__ void store256_half_C2R_ax0( float2 *z, float *data, float2 *data129, int sx, int tid )
{
    #pragma unroll
    for( int i = 0; i < (radix>>1); i++ )
	data[(i*sx)<<7] = z[rev<radix>(i)].x;

    if (tid == 0) data129[0] = make_float2( 0.0f, 0.0f);
}

template<int radix> inline __device__ void store4096_half_C2R_ax1( float2 *z, float *data, float2 *data1025, int sx, int tid )
{
    #pragma unroll
    for( int i = 0; i < radix>>1; i++ )
	data[(i*sx)] = z[rev<radix>(i)].x;

    if (tid == 0) data1025[0] = make_float2( 0.0f, 0.0f);
}

template<int radix> inline __device__ void store256_half_C2R_ax2( float2 *z, float *data, float2 *data129, int sx, int tid )
{
    #pragma unroll
    for( int i = 0; i < radix>>1; i++ )
	data[(i*sx)<<14] = z[rev<radix>(i)].x;

    if (tid == 0) data129[0] = make_float2( 0.0f, 0.0f);
}

template<int radix> inline __device__ void store256_half_C2R_ax3( float2 *z, float *data, float2 *data129, int sx, int tid )
{
    #pragma unroll
    for( int i = 0; i < radix>>1; i++ )
	data[(i*sx)<<21] = z[rev<radix>(i)].x;

    if (tid == 0) data129[0] = make_float2( 0.0f, 0.0f);
}
/////////////////
//     C2C     //
/////////////////

////    LOADS C2C    ////

// axSplit 0

inline __device__ void load256_half_C2C_ax1_axSplit0( int radix, float2 *z, float *re, float *im, int sx )
{
    for( int i = 0; i < radix; i++ )
        if ( i < (radix>>1) )
        {
            z[i].x = re[i*sx];
            z[i].y = im[i*sx];
        }
        else
            z[i] = make_float2( 0.0f, 0.0f );    
}

inline __device__ void load256_half_C2C_129_ax1_axSplit0( int radix, float2 *z, float2 *data129, int sx )
{
    for( int i = 0; i < radix; i++ )
        if ( i < (radix>>1) )
            z[i] = data129[i*sx];
        else
            z[i] = make_float2( 0.0f, 0.0f );
}

inline __device__ void load256_half_C2C_ax2_axSplit0( int radix, float2 *z, float *re, float *im, int sx )
{
    for( int i = 0; i < radix; i++ )
        if ( i < (radix>>1) )
        {
            z[i].x = re[(i*sx)<<14];	
            z[i].y = im[(i*sx)<<14];	            
        }
        else
            z[i] = make_float2( 0.0f, 0.0f );
}

inline __device__ void load256_half_C2C_129_ax2_axSplit0( int radix, float2 *z, float2 *data129, int sx )
{
    for( int i = 0; i < radix; i++ )
        if ( i < (radix>>1) )
            z[i] = data129[(i*sx)<<7];
        else
            z[i] = make_float2( 0.0f, 0.0f );
}

inline __device__ void load256_half_C2C_ax3_axSplit0( int radix, float2 *z, float *re, float *im, int sx )
{
    for( int i = 0; i < radix; i++ )
        if ( i < (radix>>1) )
        {
            z[i].x = re[(i*sx)<<21];	
            z[i].y = im[(i*sx)<<21];	            
        }
        else
            z[i] = make_float2( 0.0f, 0.0f );
}

inline __device__ void load256_half_C2C_129_ax3_axSplit0( int radix, float2 *z, float2 *data129, int sx )
{
    for( int i = 0; i < radix; i++ )
        if ( i < (radix>>1) )
            z[i] = data129[(i*sx)<<14];
        else
            z[i] = make_float2( 0.0f, 0.0f );
}

// axSplit 1

inline __device__ void load256_half_C2C_ax0_axSplit1( int radix, float2 *z, float *re, float *im, int sx )
{
    for( int i = 0; i < radix; i++ )
        if ( i < (radix>>1) )
        {
            z[i].x = re[(i*sx)<<7];
            z[i].y = im[(i*sx)<<7];
        }
        else
            z[i] = make_float2( 0.0f, 0.0f );
}

inline __device__ void load256_half_C2C_129_ax0_axSplit1( int radix, float2 *z, float2 *data129, int sx )
{
    for( int i = 0; i < radix; i++ )
        if ( i < (radix>>1) )
            z[i] = data129[i*sx];
        else
            z[i] = make_float2( 0.0f, 0.0f );
}

inline __device__ void load256_half_C2C_ax2_axSplit1( int radix, float2 *z, float *re, float *im, int sx )
{
    for( int i = 0; i < radix; i++ )
        if ( i < (radix>>1) )
        {
            z[i].x = re[(i*sx)<<14];	
            z[i].y = im[(i*sx)<<14];	            
        }
        else
            z[i] = make_float2( 0.0f, 0.0f );
}

inline __device__ void load256_half_C2C_129_ax2_axSplit1( int radix, float2 *z, float2 *data129, int sx )
{
    for( int i = 0; i < radix; i++ )
        if ( i < (radix>>1) )
            z[i] = data129[(i*sx)<<7];
        else
            z[i] = make_float2( 0.0f, 0.0f );
}

inline __device__ void load256_half_C2C_ax3_axSplit1( int radix, float2 *z, float *re, float *im, int sx )
{
    for( int i = 0; i < radix; i++ )
        if ( i < (radix>>1) )
        {
            z[i].x = re[(i*sx)<<21];	
            z[i].y = im[(i*sx)<<21];	            
        }
        else
            z[i] = make_float2( 0.0f, 0.0f );
}

inline __device__ void load256_half_C2C_129_ax3_axSplit1( int radix, float2 *z, float2 *data129, int sx )
{
    for( int i = 0; i < radix; i++ )
        if ( i < (radix>>1) )
            z[i] = data129[(i*sx)<<14];
        else
            z[i] = make_float2( 0.0f, 0.0f );
}

// axSplit 2

inline __device__ void load256_half_C2C_ax0_axSplit2( int radix, float2 *z, float *re, float *im, int sx )
{
    for( int i = 0; i < radix; i++ )
        if ( i < (radix>>1) )
        {
            z[i].x = re[(i*sx)<<7];
            z[i].y = im[(i*sx)<<7];
        }
        else
            z[i] = make_float2( 0.0f, 0.0f );
}

inline __device__ void load256_half_C2C_129_ax0_axSplit2( int radix, float2 *z, float2 *data129, int sx )
{
    for( int i = 0; i < radix; i++ )
        if ( i < (radix>>1) )
            z[i] = data129[(i*sx)<<7];
        else
            z[i] = make_float2( 0.0f, 0.0f );        
}

inline __device__ void load256_half_C2C_ax1_axSplit2( int radix, float2 *z, float *re, float *im, int sx )
{
    for( int i = 0; i < radix; i++ )
        if ( i < (radix>>1) )
        {
            z[i].x = re[i*sx];
            z[i].y = im[i*sx];
        }
        else
            z[i] = make_float2( 0.0f, 0.0f );  
}

inline __device__ void load256_half_C2C_129_ax1_axSplit2( int radix, float2 *z, float2 *data129, int sx )
{
    for( int i = 0; i < radix; i++ )
        if ( i < (radix>>1) )
            z[i] = data129[i*sx];
        else
            z[i] = make_float2( 0.0f, 0.0f );  
}

inline __device__ void load256_half_C2C_ax3_axSplit2( int radix, float2 *z, float *re, float *im, int sx )
{
    for( int i = 0; i < radix; i++ )
        if ( i < (radix>>1) )
        {
            z[i].x = re[(i*sx)<<21];
            z[i].y = im[(i*sx)<<21];
        }
        else
            z[i] = make_float2( 0.0f, 0.0f );
}

inline __device__ void load256_half_C2C_129_ax3_axSplit2( int radix, float2 *z, float2 *data129, int sx )
{
    for( int i = 0; i < radix; i++ )
        if ( i < (radix>>1) )
            z[i] = data129[(i*sx)<<14];
        else
            z[i] = make_float2( 0.0f, 0.0f );
}

// axSplit 3

inline __device__ void load256_half_C2C_ax0_axSplit3( int radix, float2 *z, float *re, float *im, int sx )
{
    for( int i = 0; i < radix; i++ )
        if ( i < (radix>>1) )
        {
            z[i].x = re[(i*sx)<<7];
            z[i].y = im[(i*sx)<<7];
        }
        else
            z[i] = make_float2( 0.0f, 0.0f );        
}

inline __device__ void load256_half_C2C_129_ax0_axSplit3( int radix, float2 *z, float2 *data129, int sx )
{
    for( int i = 0; i < radix; i++ )
        if ( i < (radix>>1) )
            z[i] = data129[(i*sx)<<7];
        else
            z[i] = make_float2( 0.0f, 0.0f );
}

inline __device__ void load256_half_C2C_ax1_axSplit3( int radix, float2 *z, float *re, float *im, int sx )
{
    for( int i = 0; i < radix; i++ )
        if ( i < (radix>>1) )
        {
            z[i].x = re[i*sx];
            z[i].y = im[i*sx];
        }
        else
            z[i] = make_float2( 0.0f, 0.0f );
}

inline __device__ void load256_half_C2C_129_ax1_axSplit3( int radix, float2 *z, float2 *data129, int sx )
{
    for( int i = 0; i < radix; i++ )
        if ( i < (radix>>1) )
            z[i] = data129[i*sx];
        else
            z[i] = make_float2( 0.0f, 0.0f );
}

inline __device__ void load256_half_C2C_ax2_axSplit3( int radix, float2 *z, float *re, float *im, int sx )
{
    for( int i = 0; i < radix; i++ )
        if ( i < (radix>>1) )
        {
            z[i].x = re[(i*sx)<<14];
            z[i].y = im[(i*sx)<<14];
        }
        else
            z[i] = make_float2( 0.0f, 0.0f );
}

inline __device__ void load256_half_C2C_129_ax2_axSplit3( int radix, float2 *z, float2 *data129, int sx )
{
    for( int i = 0; i < radix; i++ )
        if ( i < (radix>>1) )
            z[i] = data129[(i*sx)<<14];
        else
            z[i] = make_float2( 0.0f, 0.0f );
}

////   STORES C2C   ////

// axSplit 0

template<int radix> inline __device__ void store256_half_C2C_ax1_axSplit0( float2 *z, float *re, float *im, int sx )
{
    #pragma unroll
    for( int i = 0; i < (radix>>1); i++ )
    {
    	re[i*sx] = z[rev<radix>(i)].x;
    	im[i*sx] = z[rev<radix>(i)].y;
    }
}

template<int radix> inline __device__ void store256_half_C2C_129_ax1_axSplit0( float2 *z, float2 *data129, int sx )
{
    #pragma unroll
    for( int i = 0; i < (radix>>1); i++ )
    	data129[i*sx] = z[rev<radix>(i)];
}

template<int radix> inline __device__ void store256_half_C2C_ax2_axSplit0( float2 *z, float *re, float *im, int sx )
{
    #pragma unroll
    for( int i = 0; i < (radix>>1); i++ )
    {
    	re[(i*sx)<<14] = z[rev<radix>(i)].x;
    	im[(i*sx)<<14] = z[rev<radix>(i)].y;
    }	
}

template<int radix> inline __device__ void store256_half_C2C_129_ax2_axSplit0( float2 *z, float2 *data129, int sx )
{
    #pragma unroll
    for ( int i = 0; i < (radix>>1); i++ )
    	data129[(i*sx)<<7] = z[rev<radix>(i)];  
}

template<int radix> inline __device__ void store256_half_C2C_ax3_axSplit0( float2 *z, float *re, float *im, int sx )
{
    #pragma unroll
    for( int i = 0; i < (radix>>1); i++ )
    {
    	re[(i*sx)<<21] = z[rev<radix>(i)].x;
    	im[(i*sx)<<21] = z[rev<radix>(i)].y;
    }	
}

template<int radix> inline __device__ void store256_half_C2C_129_ax3_axSplit0( float2 *z, float2 *data129, int sx )
{
    #pragma unroll
    for ( int i = 0; i < (radix>>1); i++ )
    	data129[(i*sx)<<14] = z[rev<radix>(i)];  
}

// axSplit 1

template<int radix> inline __device__ void store256_half_C2C_ax0_axSplit1( float2 *z, float *re, float *im, int sx )
{
    #pragma unroll
    for( int i = 0; i < (radix>>1); i++ )
    {
    	re[(i*sx)<<7] = z[rev<radix>(i)].x;
    	im[(i*sx)<<7] = z[rev<radix>(i)].y;
    }
}

template<int radix> inline __device__ void store256_half_C2C_129_ax0_axSplit1( float2 *z, float2 *data129, int sx )
{
    #pragma unroll
    for( int i = 0; i < (radix>>1); i++ )
    	data129[i*sx] = z[rev<radix>(i)];
}

template<int radix> inline __device__ void store256_half_C2C_ax2_axSplit1( float2 *z, float *re, float *im, int sx )
{
    #pragma unroll
    for( int i = 0; i < (radix>>1); i++ )
    {
    	re[(i*sx)<<14] = z[rev<radix>(i)].x;
    	im[(i*sx)<<14] = z[rev<radix>(i)].y;
    }	
}

template<int radix> inline __device__ void store256_half_C2C_129_ax2_axSplit1( float2 *z, float2 *data129, int sx )
{
    #pragma unroll
    for ( int i = 0; i < (radix>>1); i++ )
    	data129[(i*sx)<<7] = z[rev<radix>(i)];  
}

template<int radix> inline __device__ void store256_half_C2C_ax3_axSplit1( float2 *z, float *re, float *im, int sx )
{
    #pragma unroll
    for( int i = 0; i < (radix>>1); i++ )
    {
    	re[(i*sx)<<21] = z[rev<radix>(i)].x;
    	im[(i*sx)<<21] = z[rev<radix>(i)].y;
    }	
}

template<int radix> inline __device__ void store256_half_C2C_129_ax3_axSplit1( float2 *z, float2 *data129, int sx )
{
    #pragma unroll
    for ( int i = 0; i < (radix>>1); i++ )
    	data129[(i*sx)<<14] = z[rev<radix>(i)];  
}

// axSplit 2

template<int radix> inline __device__ void store256_half_C2C_ax0_axSplit2( float2 *z, float *re, float *im, int sx )
{
    #pragma unroll
    for( int i = 0; i < (radix>>1); i++ )
    {
    	re[(i*sx)<<7] = z[rev<radix>(i)].x;
    	im[(i*sx)<<7] = z[rev<radix>(i)].y;
    }
}

template<int radix> inline __device__ void store256_half_C2C_129_ax0_axSplit2( float2 *z, float2 *data129, int sx )
{
    #pragma unroll
    for( int i = 0; i < (radix>>1); i++ )
    	data129[(i*sx)<<7] = z[rev<radix>(i)];
}

template<int radix> inline __device__ void store256_half_C2C_ax1_axSplit2( float2 *z, float *re, float *im, int sx )
{
    #pragma unroll
    for( int i = 0; i < (radix>>1); i++ )
    {
    	re[i*sx] = z[rev<radix>(i)].x;
    	im[i*sx] = z[rev<radix>(i)].y;
    }
}

template<int radix> inline __device__ void store256_half_C2C_129_ax1_axSplit2( float2 *z, float2 *data129, int sx )
{
    #pragma unroll
    for( int i = 0; i < (radix>>1); i++ )
    	data129[i*sx] = z[rev<radix>(i)];
}

template<int radix> inline __device__ void store256_half_C2C_ax3_axSplit2( float2 *z, float *re, float *im, int sx )
{
    #pragma unroll
    for( int i = 0; i < (radix>>1); i++ )
    {
    	re[(i*sx)<<21] = z[rev<radix>(i)].x;
    	im[(i*sx)<<21] = z[rev<radix>(i)].y;
    }
}

template<int radix> inline __device__ void store256_half_C2C_129_ax3_axSplit2( float2 *z, float2 *data129, int sx )
{
    #pragma unroll
    for( int i = 0; i < (radix>>1); i++ )
    	data129[(i*sx)<<14] = z[rev<radix>(i)];
}

// axSplit 3

template<int radix> inline __device__ void store256_half_C2C_ax0_axSplit3( float2 *z, float *re, float *im, int sx )
{
    #pragma unroll
    for( int i = 0; i < (radix>>1); i++ )
    {
    	re[(i*sx)<<7] = z[rev<radix>(i)].x;
    	im[(i*sx)<<7] = z[rev<radix>(i)].y;
    }
}

template<int radix> inline __device__ void store256_half_C2C_129_ax0_axSplit3( float2 *z, float2 *data129, int sx )
{
    #pragma unroll
    for( int i = 0; i < (radix>>1); i++ )
    	data129[(i*sx)<<7] = z[rev<radix>(i)];
}

template<int radix> inline __device__ void store256_half_C2C_ax1_axSplit3( float2 *z, float *re, float *im, int sx )
{
    #pragma unroll
    for( int i = 0; i < (radix>>1); i++ )
    {
    	re[i*sx] = z[rev<radix>(i)].x;
    	im[i*sx] = z[rev<radix>(i)].y;
    }
}

template<int radix> inline __device__ void store256_half_C2C_129_ax1_axSplit3( float2 *z, float2 *data129, int sx )
{
    #pragma unroll
    for( int i = 0; i < (radix>>1); i++ )
    	data129[i*sx] = z[rev<radix>(i)];
}

template<int radix> inline __device__ void store256_half_C2C_ax2_axSplit3( float2 *z, float *re, float *im, int sx )
{
    #pragma unroll
    for( int i = 0; i < (radix>>1); i++ )
    {
    	re[(i*sx)<<14] = z[rev<radix>(i)].x;
    	im[(i*sx)<<14] = z[rev<radix>(i)].y;
    }
}

template<int radix> inline __device__ void store256_half_C2C_129_ax2_axSplit3( float2 *z, float2 *data129, int sx )
{
    #pragma unroll
    for( int i = 0; i < (radix>>1); i++ )
    	data129[(i*sx)<<14] = z[rev<radix>(i)];
}

// *****************************************
// ***** FRACTIONAL FOURIER TRANSFORM ******
// *****************************************

inline __device__ void gen_y( int radix, float2 *z, float delta, int tid )
{
    for ( int j = 0; j < radix/2; j++ )
        z[j] = operator_mul_zz( z[j], exp_i( M_PI*(tid + M_base*j)*delta* ( M_points - (tid + M_base*j)) ) );
}

inline __device__ void gen_z( int radix, float2 *z, float delta, int segment, int tid )
{
    for ( int j = 0; j < radix; j++ )
        if ( j < radix/2 )
            z[j] = exp_i( M_PI* powf(tid + M_base*j + segment, 2) *delta );
        else
            z[j] = exp_i( M_PI* powf(tid + M_base*j - 2*M_points + segment, 2) *delta );
}

template<int radix> inline __device__ void complexMulYZ( float2 *w, float2 *y, float2 *z )
{
    for ( int i = 0; i < radix; i++ ) 
        w[rev<radix>(i)] = operator_mul_zz( y[i], z[i] );
}

template<int radix> inline __device__ void complexMulExpZ( float2 *y, float2 *z, float dx, float delta, int segment, int tid )
{
    for ( int i = 0; i < radix; i++ )
        y[rev<radix>(i)] = operator_mul_zz( z[rev<radix>(i)], operator_mul_za( exp_i( M_PI* ( (tid + M_base*i - M_points/2 + segment)*M_points*delta - powf(tid + M_base*i + segment, 2) *delta ) ), dx ) );        
}

// ************************

__device__ void GENERAL_FAFT4096( float2 *y, float dx, float delta, int segment, int tid )
{
    __shared__ float smdata[4113];

    gen_y( 16, y, delta, tid);
    
    FFT16( y );
    twiddle<16>( y, tid, 4096 );

    storeSharedx<16>( y, &smdata[tid], 257 );
    __syncthreads();    
    loadSharedx( 16, y, &smdata[257*(tid&15) + (tid>>4)], 16 );	
    __syncthreads();
    storeSharedy<16>( y, &smdata[tid], 257 );
    __syncthreads();
    loadSharedy( 16, y, &smdata[257*(tid&15) + (tid>>4)], 16 );	
    
    FFT16( y );
    twiddle<16>( y, (tid>>4), 256 );
    
    storeSharedx<16>( y, &smdata[tid], 257 );
    __syncthreads();
    loadSharedx( 16, y, &smdata[257*(tid>>4) + (tid&15)], 16 );	
    __syncthreads();
    storeSharedy<16>( y, &smdata[tid], 257 );    
    __syncthreads();
    loadSharedy( 16, y, &smdata[257*(tid>>4) + (tid&15)], 16 );	

    FFT16( y );
        
    // ************************** Z ****************************
    
    float2 z[16];
    
    gen_z( 16, z, delta, segment, tid );
    
    FFT16( z );
    twiddle<16>( z, tid, 4096 );

    storeSharedx<16>( z, &smdata[tid], 257 );
    __syncthreads();    
    loadSharedx( 16, z, &smdata[257*(tid&15) + (tid>>4)], 16 );	
    __syncthreads();
    storeSharedy<16>( z, &smdata[tid], 257 );
    __syncthreads();
    loadSharedy( 16, z, &smdata[257*(tid&15) + (tid>>4)], 16 );	
    
    FFT16( z );
    twiddle<16>( z, (tid>>4), 256 );
    
    storeSharedx<16>( z, &smdata[tid], 257 );
    __syncthreads();
    loadSharedx( 16, z, &smdata[257*(tid>>4) + (tid&15)], 16 );	
    __syncthreads();
    storeSharedy<16>( z, &smdata[tid], 257 );    
    __syncthreads();
    loadSharedy( 16, z, &smdata[257*(tid>>4) + (tid&15)], 16 );	

    FFT16( z );

    float2 w[16];
    complexMulYZ<16>( w, y, z );
    
    // ********************  Inverse ****************************

    IFFT16( w );
    itwiddle<16>( w, tid, 4096 );

    storeSharedx<16>( w, &smdata[tid], 257 );
    __syncthreads();    
    loadSharedx( 16, w, &smdata[257*(tid&15) + (tid>>4)], 16 );	
    __syncthreads();
    storeSharedy<16>( w, &smdata[tid], 257 );
    __syncthreads();
    loadSharedy( 16, w, &smdata[257*(tid&15) + (tid>>4)], 16 );	
    
    IFFT16( w );
    itwiddle<16>( w, (tid>>4), 256 );
    
    storeSharedx<16>( w, &smdata[tid], 257 );
    __syncthreads();
    loadSharedx( 16, w, &smdata[257*(tid>>4) + (tid&15)], 16 );	
    __syncthreads();
    storeSharedy<16>( w, &smdata[tid], 257 );    
    __syncthreads();
    loadSharedy( 16, w, &smdata[257*(tid>>4) + (tid&15)], 16 );	

    IFFT16( w );    
    
    complexMulExpZ<16>( y, w, dx, delta, segment, tid );
}

