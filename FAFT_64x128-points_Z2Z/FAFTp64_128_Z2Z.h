// Fast Accurate Fourier Transform (FAFT) was written by Oscar R. Cabrera L.
// Contributors: Renan Cabrera, Denys I. Bondar.
// Copyright (c) 2016
// All rights reserved.
	
#pragma once

#define _USE_MATH_DEFINES
#include <math.h>

inline __device__ double2 operator_mul_zz( double2 a, double2 b ){ return make_double2( a.x*b.x-a.y*b.y, a.x*b.y+a.y*b.x ); }
inline __device__ double2 operator_mul_za( double2 a, double b  ){ return make_double2( b*a.x, b*a.y ); }
inline __device__ double2 operator_div_za( double2 a, double b  ){ return make_double2( a.x/b, a.y/b ); }
inline __device__ double2 operator_plu_zz( double2 a, double2 b ){ return make_double2( a.x + b.x, a.y + b.y ); }
inline __device__ double2 operator_min_zz( double2 a, double2 b ){ return make_double2( a.x - b.x, a.y - b.y ); }

#define cos_pi_8  0.9238795325112867
#define sin_pi_8  0.3826834323650898
#define sqrt_5_4  0.5590169943749474
#define sin_2pi_5 0.9510565162951535
#define sin_pi_5  0.5877852522924731

#define imag      make_double2( 0.0,  1.0 )
#define imag_neg  make_double2( 0.0, -1.0 )

#define exp_1_16  make_double2(  cos_pi_8, -sin_pi_8 )
#define exp_3_16  make_double2(  sin_pi_8, -cos_pi_8 )
#define exp_5_16  make_double2( -sin_pi_8, -cos_pi_8 )
#define exp_7_16  make_double2( -cos_pi_8, -sin_pi_8 )
#define exp_9_16  make_double2( -cos_pi_8,  sin_pi_8 )
#define exp_1_8   make_double2(  1.0, -1.0 )
#define exp_1_4   make_double2(  0.0, -1.0 )
#define exp_3_8   make_double2( -1.0, -1.0 )

#define iexp_1_16  make_double2(  cos_pi_8,  sin_pi_8 )
#define iexp_3_16  make_double2(  sin_pi_8,  cos_pi_8 )
#define iexp_5_16  make_double2( -sin_pi_8,  cos_pi_8 )
#define iexp_7_16  make_double2( -cos_pi_8,  sin_pi_8 )
#define iexp_9_16  make_double2( -cos_pi_8, -sin_pi_8 )
#define iexp_1_8   make_double2(  1.0, 1.0 )
#define iexp_1_4   make_double2(  0.0, 1.0 )
#define iexp_3_8   make_double2( -1.0, 1.0 )

#define M_points_64 64
#define M_base_64 16

#define M_points_128 128
#define M_base_128 64


inline __device__ double2 exp_i( double phi )
{
    return make_double2( (double)cos(phi), (double)sin(phi) );
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
inline __device__ void FFT2( double2 *z0, double2 *z1 )
{ 
    double2 t0 = *z0;
    
    *z0 = operator_plu_zz( t0, *z1 ); 
    *z1 = operator_min_zz( t0, *z1 );
}

inline __device__ void FFT4( double2 *z0, double2 *z1, double2 *z2, double2 *z3 )
{
    FFT2( z0, z2 );
    FFT2( z1, z3 );
    *z3 = operator_mul_zz( *z3, exp_1_4 );
    FFT2( z0, z1 );
    FFT2( z2, z3 );
    
}

inline __device__ void IFFT4( double2 *z0, double2 *z1, double2 *z2, double2 *z3 )
{
    IFFT2( z0, z2 );
    IFFT2( z1, z3 );
    *z3 = operator_mul_zz( *z3, iexp_1_4 );
    IFFT2( z0, z1 );
    IFFT2( z2, z3 );
}

inline __device__ void  FFT2vec( double2 *z ) {  FFT2( &z[0], &z[1] ); }
inline __device__ void IFFT2vec( double2 *z ) { IFFT2( &z[0], &z[1] ); }
inline __device__ void  FFT4vec( double2 *z ) {  FFT4( &z[0], &z[1], &z[2], &z[3] ); }
inline __device__ void IFFT4vec( double2 *z ) { IFFT4( &z[0], &z[1], &z[2], &z[3] ); }

inline __device__ void FFT8( double2 *z )
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

inline __device__ void IFFT8( double2 *z )
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

inline __device__  void FFT5( double2 *z)
{
    double2 t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11;
    
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

inline __device__  void IFFT5( double2 *z)
{
    double2 t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11;
    
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

inline __device__ void FFT16( double2 *z )
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

inline __device__ void IFFT16( double2 *z )
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
inline __device__ void mulByMinusOne( int radix, double2 *z, size_t sector, int tid )
{
    for ( int i = 0; i < radix; i++ )
        z[i] = operator_mul_za( z[i], pow( (double)(-1), (double)( ( (sector/160) + (sector % 160) + tid ) & 1) ));   
}
*/

inline __device__ void normalize( int radix, double2 *z, double normFactor )
{
    for ( int i = 0; i < radix; i++ )
        z[i] = operator_div_za( z[i], normFactor );
}

//////////////////
//   TWIDDLES   //
//////////////////

template<int radix> inline __device__ void twiddle( double2 *z, int i, int n )
{
    for ( int j = 1; j < radix; j++ )
        z[j] = operator_mul_zz( z[j], exp_i((-2*M_PI*rev<radix>( j )/n)*i) );
}

template<int radix> inline __device__ void itwiddle( double2 *z, int i, int n )
{
    for ( int j = 1; j < radix; j++ )
        z[j] = operator_mul_zz( z[j], exp_i(( 2*M_PI*rev<radix>( j )/n)*i) );
}

///////////////////////
//   SHARED MEMORY   //
///////////////////////

inline __device__ void loadSharedx( int radix, double2 *z, double *a, int sx )
{
    for( int i = 0; i < radix; i++ )
        z[i].x = a[i*sx];
}

inline __device__ void loadSharedy( int radix, double2 *z, double *a, int sx )
{
    for( int i = 0; i < radix; i++ )
        z[i].y = a[i*sx];
}

template<int radix> inline __device__ void storeSharedx( double2 *z, double *a, int sx )
{
    #pragma unroll
    for( int i = 0; i < radix; i++ )
        a[i*sx] = z[rev<radix>( i )].x;
}

template<int radix> inline __device__ void storeSharedy( double2 *z, double *a, int sx )
{
    #pragma unroll
    for( int i = 0; i < radix; i++ )
        a[i*sx] = z[rev<radix>( i )].y;
}


/////////////////
//     Z2Z     //
/////////////////

////    LOADS Z2Z    ////

// (128, 128, 64, 64)

inline __device__ void load128_half_Z2Z_ax2( int radix, double2 *z, double2 *a, int sx )	// 0 -> 2
{
    for( int i = 0; i < radix; i++ )
        if ( i < (radix>>1) )
            z[i] = a[(i*sx)<<6];
        else
            z[i] = make_double2( 0.0, 0.0 );
}

inline __device__ void load128_half_Z2Z_ax3( int radix, double2 *z, double2 *a, int sx )	// 1 -> 3
{
    for( int i = 0; i < radix; i++ )
        if ( i < (radix>>1) )
            z[i] = a[(i*sx)];
        else
            z[i] = make_double2( 0.0, 0.0 );
}

// (64, 64, 128, 128)

inline __device__ void load128_half_Z2Z_ax1( int radix, double2 *z, double2 *a, int sx )	// 2 -> 1
{
    for( int i = 0; i < radix; i++ )
        if ( i < (radix>>1) )
            z[i] = a[(i*sx)<<14];
        else
            z[i] = make_double2( 0.0, 0.0 );
}

inline __device__ void load128_half_Z2Z_ax0( int radix, double2 *z, double2 *a, int sx )	// 3 -> 0
{
    for( int i = 0; i < radix; i++ )
        if ( i < (radix>>1) )
            z[i] = a[(i*sx)<<20];
        else
            z[i] = make_double2( 0.0, 0.0 );
}

////    STORES Z2Z    ////

// (128, 128, 64, 64)

template<int radix> inline __device__ void store128_half_Z2Z_ax2( double2 *z, double2 *a, int sx )	// 0 -> 2
{
    #pragma unroll
    for( int i = 0; i < (radix>>1); i++ )
    	a[(i*sx)<<6] = z[rev<radix>(i)];
}

template<int radix> inline __device__ void store128_half_Z2Z_ax3( double2 *z, double2 *a, int sx )	// 1 -> 3
{
    #pragma unroll
    for( int i = 0; i < (radix>>1); i++ )
    	a[(i*sx)] = z[rev<radix>(i)];
}

// (64, 64, 128, 128)

template<int radix> inline __device__ void store128_half_Z2Z_ax1( double2 *z, double2 *a, int sx )	// 2 -> 1
{
    #pragma unroll
    for( int i = 0; i < (radix>>1); i++ )
    	a[(i*sx)<<14] = z[rev<radix>(i)];
}

template<int radix> inline __device__ void store128_half_Z2Z_ax0( double2 *z, double2 *a, int sx )	// 3 -> 0
{
    #pragma unroll
    for( int i = 0; i < (radix>>1); i++ )
    	a[(i*sx)<<20] = z[rev<radix>(i)];
}


////    LOADS Z2Z    ////

// (64, 64, 128, 128)

inline __device__ void load256_half_Z2Z_ax2( int radix, double2 *z, double2 *a, int sx )	// 0 -> 2
{
    for( int i = 0; i < radix; i++ )
        if ( i < (radix>>1) )
            z[i] = a[(i*sx)<<7];
        else
            z[i] = make_double2( 0.0, 0.0 );
}

inline __device__ void load256_half_Z2Z_ax3( int radix, double2 *z, double2 *a, int sx )	// 1 -> 3
{
    for( int i = 0; i < radix; i++ )
        if ( i < (radix>>1) )
            z[i] = a[(i*sx)];
        else
            z[i] = make_double2( 0.0, 0.0 );
}

// (128, 128, 64, 64)

inline __device__ void load256_half_Z2Z_ax1( int radix, double2 *z, double2 *a, int sx )	// 2 -> 1
{
    for( int i = 0; i < radix; i++ )
        if ( i < (radix>>1) )
            z[i] = a[(i*sx)<<12];
        else
            z[i] = make_double2( 0.0, 0.0 );
}

inline __device__ void load256_half_Z2Z_ax0( int radix, double2 *z, double2 *a, int sx )	// 3 -> 0
{
    for( int i = 0; i < radix; i++ )
        if ( i < (radix>>1) )
            z[i] = a[(i*sx)<<19];
        else
            z[i] = make_double2( 0.0, 0.0 );
}

////    STORES Z2Z    ////

// (64, 64, 128, 128)

template<int radix> inline __device__ void store256_half_Z2Z_ax2( double2 *z, double2 *a, int sx )	// 0 -> 2
{
    #pragma unroll
    for( int i = 0; i < (radix>>1); i++ )
    	a[(i*sx)<<7] = z[rev<radix>(i)];
}

template<int radix> inline __device__ void store256_half_Z2Z_ax3( double2 *z, double2 *a, int sx )	// 1 -> 3
{
    #pragma unroll
    for( int i = 0; i < (radix>>1); i++ )
    	a[(i*sx)] = z[rev<radix>(i)];
}

// (128, 128, 64, 64)

template<int radix> inline __device__ void store256_half_Z2Z_ax1( double2 *z, double2 *a, int sx )	// 2 -> 1
{
    #pragma unroll
    for( int i = 0; i < (radix>>1); i++ )
    	a[(i*sx)<<12] = z[rev<radix>(i)];
}

template<int radix> inline __device__ void store256_half_Z2Z_ax0( double2 *z, double2 *a, int sx )	// 3 -> 0
{
    #pragma unroll
    for( int i = 0; i < (radix>>1); i++ )
    	a[(i*sx)<<19] = z[rev<radix>(i)];
}

// *****************************************
// ***** FRACTIONAL FOURIER TRANSFORM ******
// *****************************************

inline __device__ void gen_y( int radix, double2 *z, double delta, int tid, int M_points, int M_base )
{
    for ( int j = 0; j < radix/2; j++ )
        z[j] = operator_mul_zz( z[j], exp_i( M_PI*(tid + M_base*j)*delta* ( M_points - (tid + M_base*j)) ) );
}

inline __device__ void gen_z( int radix, double2 *z, double delta, int segment, int tid, int M_points, int M_base )
{
    for ( int j = 0; j < radix; j++ )
        if ( j < radix/2 )
            z[j] = exp_i( M_PI* (tid + M_base*j + segment)*(tid + M_base*j + segment) *delta );
        else
            z[j] = exp_i( M_PI* (tid + M_base*j - 2*M_points + segment)*(tid + M_base*j - 2*M_points + segment) *delta );
}

template<int radix> inline __device__ void complexMulYZ( double2 *w, double2 *y, double2 *z )
{
    for ( int i = 0; i < radix; i++ ) 
        w[rev<radix>(i)] = operator_mul_zz( y[i], z[i] );
}

template<int radix> inline __device__ void complexMulExpZ( double2 *y, double2 *z, double dx, double delta, int segment, int tid, int M_points, int M_base )
{
    for ( int i = 0; i < radix; i++ )
        y[rev<radix>(i)] = operator_mul_zz( z[rev<radix>(i)], operator_mul_za( exp_i( M_PI* ( (tid + M_base*i - M_points/2 + segment)*M_points*delta - (tid + M_base*i + segment)*(tid + M_base*i + segment) *delta ) ), dx ) );                
}

// ****   GENERAL FAFT 128   ****

__device__ void GENERAL_FAFT128( double2 *y, double dx, double delta, int segment, int tid )
{
    __shared__ double smdatax[162];
    __shared__ double smdatay[162];
    
    gen_y( 8, y, delta, tid, M_points_64, M_base_64 );
    
    FFT8( y );
    twiddle<8>( y, tid, 128 );
    
    storeSharedx<8>( y, &smdatax[tid], 18 );
    storeSharedy<8>( y, &smdatay[tid], 18 );
    
    __syncthreads();
    
    if (tid < 8)
    {
	loadSharedx( 16, y, &smdatax[18*tid], 1 );
	loadSharedy( 16, y, &smdatay[18*tid], 1 );
	
    	FFT16( y );
    }

    __syncthreads();
    
    // ****** Z ******
    
    double2 z[16];
    
    gen_z( 8, z, delta, segment, tid, M_points_64, M_base_64 );
    
    FFT8( z );
    twiddle<8>( z, tid, 128 );
    
    storeSharedx<8>( z, &smdatax[tid], 18 );
    storeSharedy<8>( z, &smdatay[tid], 18 );
    
    __syncthreads();
    
    if (tid < 8)
    {
        double2 w[16];
        
	loadSharedx( 16, z, &smdatax[18*tid], 1 );
	loadSharedy( 16, z, &smdatay[18*tid], 1 );
	
    	FFT16( z );

    	complexMulYZ<16>( w, y, z );
	
	// ****** Inverse ******
	
	IFFT16( w );    	
    	itwiddle<16>( w, tid, 128 );
    	
    	storeSharedx<16>( w, &smdatax[tid], 9 );
    	storeSharedy<16>( w, &smdatay[tid], 9 );
    }
    
    __syncthreads(); 
	
    loadSharedx( 8, z, &smdatax[9*tid], 1 );
    loadSharedy( 8, z, &smdatay[9*tid], 1 );	
    
    IFFT8( z );
    
    complexMulExpZ<8>( y, z, dx, delta, segment, tid, M_points_64, M_base_64 );
}


// ****   GENERAL FAFT 256   ****

__device__ void GENERAL_FAFT256( double2 *y, double dx, double delta, int segment, int tid )
{
    __shared__ double smdatax[277];
    __shared__ double smdatay[277];
    
    gen_y( 4, y, delta, tid, M_points_128, M_base_128 );
    
    FFT4vec( y );
    twiddle<4>( y, tid, 256 );
    
    storeSharedx<4>( y, &smdatax[tid], 65 );
    storeSharedy<4>( y, &smdatay[tid], 65 );
    
    __syncthreads();
    
    loadSharedx( 4, y, &smdatax[65*(tid&3) + (tid>>2)], 16 );	
    loadSharedy( 4, y, &smdatay[65*(tid&3) + (tid>>2)], 16 );	
    
    FFT4vec( y );
    twiddle<4>( y, (tid>>2), 64 );
    
    storeSharedx<4>( y, &smdatax[tid], 67 );
    storeSharedy<4>( y, &smdatay[tid], 67 );

    __syncthreads();
    
    loadSharedx( 4, y, &smdatax[67*(tid&3) + (tid>>2)], 16 );	
    loadSharedy( 4, y, &smdatay[67*(tid&3) + (tid>>2)], 16 );	

    FFT4vec( y );
    twiddle<4>( y, (tid>>4), 16 );

    storeSharedx<4>( y, &smdatax[tid], 69 );
    storeSharedy<4>( y, &smdatay[tid], 69 );

    __syncthreads();

    loadSharedx( 4, y, &smdatax[69*(tid>>4) + (tid&3)*4 + ((tid&15)>>2)], 16 );
    loadSharedy( 4, y, &smdatay[69*(tid>>4) + (tid&3)*4 + ((tid&15)>>2)], 16 );

    FFT4vec( y );
    
    // ************************** ZZZ ****************************

    double2 z[4];

    gen_z( 4, z, delta, segment, tid, M_points_128, M_base_128 );

    FFT4vec( z );
    twiddle<4>( z, tid, 256 );

    storeSharedx<4>( z, &smdatax[tid], 65 );
    storeSharedy<4>( z, &smdatay[tid], 65 );

    __syncthreads();

    loadSharedx( 4, z, &smdatax[65*(tid&3) + (tid>>2)], 16 );	
    loadSharedy( 4, z, &smdatay[65*(tid&3) + (tid>>2)], 16 );	

    FFT4vec( z );
    twiddle<4>( z, (tid>>2), 64 );

    storeSharedx<4>( z, &smdatax[tid], 67 );
    storeSharedy<4>( z, &smdatay[tid], 67 );

    __syncthreads();

    loadSharedx( 4, z, &smdatax[67*(tid&3) + (tid>>2)], 16 );	
    loadSharedy( 4, z, &smdatay[67*(tid&3) + (tid>>2)], 16 );	

    FFT4vec( z );
    twiddle<4>( z, (tid>>4), 16 );

    storeSharedx<4>( z, &smdatax[tid], 69 );
    storeSharedy<4>( z, &smdatay[tid], 69 );

    __syncthreads();

    loadSharedx( 4, z, &smdatax[69*(tid>>4) + (tid&3)*4 + ((tid&15)>>2)], 16 );
    loadSharedy( 4, z, &smdatay[69*(tid>>4) + (tid&3)*4 + ((tid&15)>>2)], 16 );

    FFT4vec( z );

    double2 w[4];

    complexMulYZ<4>( w, y, z );

    // ********************  inverse ****************************

    IFFT4vec( w );
    itwiddle<4>( w, tid, 256 );

    storeSharedx<4>( w, &smdatax[tid], 65 );
    storeSharedy<4>( w, &smdatay[tid], 65 );

    __syncthreads();

    loadSharedx( 4, w, &smdatax[65*(tid&3) + (tid>>2)], 16 );	
    loadSharedy( 4, w, &smdatay[65*(tid&3) + (tid>>2)], 16 );	

    IFFT4vec( w );
    itwiddle<4>( w, (tid>>2), 64 );

    storeSharedx<4>( w, &smdatax[tid], 67 );
    storeSharedy<4>( w, &smdatay[tid], 67 );

    __syncthreads();

    loadSharedx( 4, w, &smdatax[67*(tid&3) + (tid>>2)], 16 );	
    loadSharedy( 4, w, &smdatay[67*(tid&3) + (tid>>2)], 16 );	

    IFFT4vec( w );
    itwiddle<4>( w, (tid>>4), 16 );

    storeSharedx<4>( w, &smdatax[tid], 69 );
    storeSharedy<4>( w, &smdatay[tid], 69 );

    __syncthreads();

    loadSharedx( 4, w, &smdatax[69*(tid>>4) + (tid&3)*4 + ((tid&15)>>2)], 16 );
    loadSharedy( 4, w, &smdatay[69*(tid>>4) + (tid&3)*4 + ((tid&15)>>2)], 16 );

    IFFT4vec( w );

    complexMulExpZ<4>( y, w, dx, delta, segment, tid, M_points_128, M_base_128 );
}


