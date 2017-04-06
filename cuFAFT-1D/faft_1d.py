import numpy as np
import ctypes
from ctypes import *

import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.elementwise import ElementwiseKernel

import cufft_wrapper as cufft

class Plan_FAFT_1D_Z2Z:
    def __init__(self, x_amplitude, theta_x_amplitude, gridDIM_x ):
        
        #...........................................................
        self.gridDIM_x         = gridDIM_x
        self.x_amplitude       =       x_amplitude
        self.theta_x_amplitude = theta_x_amplitude
        
        # Phase space step size 
        self.dx    = 2*x_amplitude  /float(gridDIM_x)  

        # Ambiguity space step size
        self.dtheta_x  = 2*theta_x_amplitude /float(gridDIM_x) 

        # delta parameters
        self.delta_x  =     self.dx*self.dtheta_x/(2*np.pi)

        # Phase space range
        self.x_range   = np.linspace( -x_amplitude,   x_amplitude  -self.dx,    gridDIM_x  ) 

        # Ambiguity space range
        self.theta_x_range  = np.linspace( -theta_x_amplitude,  theta_x_amplitude -self.dtheta_x,    gridDIM_x   ) 
        
        #
        self.productBeta_FirstHalf_GPU = ElementwiseKernel(
            "pycuda::complex<double> *f, pycuda::complex<double>  *y , double delta, int gridDIM, double dx",
            "f[i] = dx*y[i]*exp(  pycuda::complex<double>( 0., M_PI*((i-gridDIM/2)*gridDIM*delta - delta*i*i)  )  ); ",
            "productBeta_First",
            preamble = "#define _USE_MATH_DEFINES")

        #
        self.getZ_FirstHalf_GPU = ElementwiseKernel(
            "pycuda::complex<double>  *z , double delta, int gridDIM",
            "z[i] = exp(  pycuda::complex<double>( 0., M_PI*delta*i*i  )  ); ",
            "getZ_First",
            preamble = "#define _USE_MATH_DEFINES")

        self.getZ_SecondHalf_GPU = ElementwiseKernel(
            "pycuda::complex<double>  *z , double delta, int gridDIM",
            "z[i] = exp(  pycuda::complex<double>( 0., M_PI*delta*(i-2*gridDIM )*(i-2*gridDIM )  ) );",
            "getZ_Second",
            preamble = "#define _USE_MATH_DEFINES")
        #
        self.productY_FirstHalf_GPU = ElementwiseKernel(
            "pycuda::complex<double>  *y, pycuda::complex<double>  *F , double delta, int gridDIM",
            "y[i] = F[i]*exp(  pycuda::complex<double>( 0. , M_PI*(i*gridDIM*delta - delta*i*i)  )  ); ",
            "productY_First",
            preamble = "#define _USE_MATH_DEFINES")

        self.productY_SecondHalf_GPU = ElementwiseKernel(
            "pycuda::complex<double>  *y, pycuda::complex<double>  *F , double delta, int gridDIM",
            "y[i] = pycuda::complex<double>(0.,0.); ",
            "productY_Second",
            preamble = "#define _USE_MATH_DEFINES")        

        #.........................................................
        
        self.plan_y_Z2Z = cufft.Plan_Z2Z( (gridDIM_x*2,) )
        
        self.z_plus_gpu    = gpuarray.zeros( gridDIM_x*2 , dtype=np.complex128 )
        self.z_minus_gpu   = gpuarray.zeros( gridDIM_x*2 , dtype=np.complex128 )
        
        self.getZ_FirstHalf_GPU ( self.z_plus_gpu, self.delta_x, gridDIM_x , range=slice(0,gridDIM_x,1)           )
        self.getZ_SecondHalf_GPU( self.z_plus_gpu, self.delta_x, gridDIM_x , range=slice(gridDIM_x,2*gridDIM_x,1) )

        self.getZ_FirstHalf_GPU ( self.z_minus_gpu, -self.delta_x, gridDIM_x , range=slice(0,gridDIM_x,1)           )
        self.getZ_SecondHalf_GPU( self.z_minus_gpu, -self.delta_x, gridDIM_x , range=slice(gridDIM_x,2*gridDIM_x,1) )
        
        cufft.fft_Z2Z(self.z_plus_gpu,  self.z_plus_gpu,  self.plan_y_Z2Z)
        cufft.fft_Z2Z(self.z_minus_gpu, self.z_minus_gpu, self.plan_y_Z2Z)
        
    
    def FAFT( self, y_gpu ):
        """
        cuFFT_Z2Z_1D( y_gpu ) performs the FAFT Fourier transform in place for the gpu array y_gpu.
        y_gpu[0:gridDIM]         contains the actual data 
        y_gpu[grdDIM,2*gridDIM]  must be allocated for internal computation
        """
        
        assert y_gpu.shape[0] == 2*self.gridDIM_x, " Input gpu array y_gpu is not 2 gridDIM.  "
        
        self.productY_FirstHalf_GPU ( y_gpu, y_gpu, self.delta_x, self.gridDIM_x, range=slice(0,self.gridDIM_x,1)           )
        self.productY_SecondHalf_GPU( y_gpu, y_gpu, self.delta_x, self.gridDIM_x, range=slice(self.gridDIM_x,2*self.gridDIM_x,1) )

        cufft.fft_Z2Z( y_gpu, y_gpu, self.plan_y_Z2Z )

        y_gpu *= self.z_plus_gpu

        cufft.ifft_Z2Z(  y_gpu, y_gpu, self.plan_y_Z2Z )

        self.productBeta_FirstHalf_GPU( y_gpu, y_gpu, self.delta_x, self.gridDIM_x, self.dx, range=slice(0,self.gridDIM_x,1)   ) 
        
        y_gpu /= np.float64(self.gridDIM_x*self.gridDIM_x)
 
    def norm_GPU(self, W_gpu):
        return gpuarray.sum( W_gpu[0:self.gridDIM_x] ).get().real*self.dx    
        
    def iFAFT( self, y_gpu ):
        """
        cuFFT_Z2Z_1D( y_gpu ) performs the FAFT Fourier transform in place for the gpu array y_gpu.
        y_gpu[0:gridDIM]         contains the actual data 
        y_gpu[grdDIM,2*gridDIM]  must be allocated for internal computation
        """
        
        assert y_gpu.shape[0] == 2*self.gridDIM_x, " Input gpu array y_gpu is not 2 gridDIM.  "
        
        self.productY_FirstHalf_GPU ( y_gpu, y_gpu, -self.delta_x, self.gridDIM_x, range=slice(0,self.gridDIM_x,1)           )
        self.productY_SecondHalf_GPU( y_gpu, y_gpu, -self.delta_x, self.gridDIM_x, range=slice(self.gridDIM_x,2*self.gridDIM_x,1) )

        cufft.fft_Z2Z( y_gpu, y_gpu, self.plan_y_Z2Z )

        y_gpu *= self.z_minus_gpu

        cufft.ifft_Z2Z(  y_gpu, y_gpu, self.plan_y_Z2Z )

        self.productBeta_FirstHalf_GPU( y_gpu, y_gpu, -self.delta_x, self.gridDIM_x, self.dx, range=slice(0,self.gridDIM_x,1)   )  