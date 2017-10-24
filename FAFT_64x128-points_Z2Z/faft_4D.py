import numpy as np
import ctypes
from ctypes import *

import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit

class Plan_FAFT_4D_Z2Z__64_64_128_128:
    def __init__(self, dir_base):
        self.dir_base = dir_base
        
        # FAFT 64-points
        self._faft64_4D = ctypes.cdll.LoadLibrary( self.dir_base+'FAFT_64_128_4D_Z2Z.so' )
        self._faft64_4D.FAFT64_4D_Z2Z.restype = int
        self._faft64_4D.FAFT64_4D_Z2Z.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double, 
                                                  ctypes.c_int, ctypes.c_int, ctypes.c_double ]
        self.faft64 = self._faft64_4D.FAFT64_4D_Z2Z

        self._faft64_4D.IFAFT64_4D_Z2Z.restype = int
        self._faft64_4D.IFAFT64_4D_Z2Z.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double, 
                                                   ctypes.c_int, ctypes.c_int, ctypes.c_double ]
        self.ifaft64 = self._faft64_4D.IFAFT64_4D_Z2Z

        # FAFT 128-points
        self._faft128_4D = ctypes.cdll.LoadLibrary( self.dir_base+'FAFT_64_128_4D_Z2Z.so' )
        self._faft128_4D.FAFT128_4D_Z2Z.restype = int
        self._faft128_4D.FAFT128_4D_Z2Z.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double, 
                                                    ctypes.c_int, ctypes.c_int, ctypes.c_double ]
        self.faft128 = self._faft128_4D.FAFT128_4D_Z2Z

        self._faft128_4D.IFAFT128_4D_Z2Z.restype = int
        self._faft128_4D.IFAFT128_4D_Z2Z.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double, 
                                                     ctypes.c_int, ctypes.c_int, ctypes.c_double ]
        self.ifaft128 = self._faft128_4D.IFAFT128_4D_Z2Z
        
    def FAFT__64_64_128_128( self, F_gpu, d, delta, segment, axes, normFactor ):
        if axes == 0 or axes == 1:
            self.faft64( int(F_gpu.gpudata), d, delta, segment, axes, normFactor )
        else:
            if axes == 2 or axes == 3:
                self.faft128(  int(F_gpu.gpudata), d, delta, segment, axes, normFactor )          
        
    def IFAFT__64_64_128_128( self, F_gpu, d, delta, segment, axes, normFactor ):
        if axes == 0 or axes == 1:
            self.ifaft64( int(F_gpu.gpudata), d, delta, segment, axes, normFactor )
        else:
            if axes == 2 or axes == 3:
                self.ifaft128(  int(F_gpu.gpudata), d, delta, segment, axes, normFactor )   
                        
    
class Plan_FAFT_4D_Z2Z__128_128_64_64:
    def __init__(self, dir_base):
        self.dir_base = dir_base
        
        # FAFT 64-points
        self._faft64_4D = ctypes.cdll.LoadLibrary( self.dir_base+'FAFT_64_128_4D_Z2Z.so' )
        self._faft64_4D.FAFT64_4D_Z2Z.restype = int
        self._faft64_4D.FAFT64_4D_Z2Z.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double, 
                                                  ctypes.c_int, ctypes.c_int, ctypes.c_double ]
        self.faft64 = self._faft64_4D.FAFT64_4D_Z2Z

        self._faft64_4D.IFAFT64_4D_Z2Z.restype = int
        self._faft64_4D.IFAFT64_4D_Z2Z.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double, 
                                                   ctypes.c_int, ctypes.c_int, ctypes.c_double ]
        self.ifaft64 = self._faft64_4D.IFAFT64_4D_Z2Z

        # FAFT 128-points
        self._faft128_4D = ctypes.cdll.LoadLibrary( self.dir_base+'FAFT_64_128_4D_Z2Z.so' )
        self._faft128_4D.FAFT128_4D_Z2Z.restype = int
        self._faft128_4D.FAFT128_4D_Z2Z.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double, 
                                                    ctypes.c_int, ctypes.c_int, ctypes.c_double ]
        self.faft128 = self._faft128_4D.FAFT128_4D_Z2Z

        self._faft128_4D.IFAFT128_4D_Z2Z.restype = int
        self._faft128_4D.IFAFT128_4D_Z2Z.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double, 
                                                     ctypes.c_int, ctypes.c_int, ctypes.c_double ]
        self.ifaft128 = self._faft128_4D.IFAFT128_4D_Z2Z
        
    def FAFT__128_128_64_64( self, F_gpu, d, delta, segment, axes, normFactor ):
        if axes == 0 or axes == 1:
            self.faft128( int(F_gpu.gpudata), d, delta, segment, axes, normFactor )
        else:
            if axes == 2 or axes == 3:
                self.faft64(  int(F_gpu.gpudata), d, delta, segment, axes, normFactor )

                        
                        
    def IFAFT__128_128_64_64( self, F_gpu, d, delta, segment, axes, normFactor ):
        if axes == 0 or axes == 1:
            self.ifaft128( int(F_gpu.gpudata), d, delta, segment, axes, normFactor )
        else:
            if axes == 2 or axes == 3:
                self.ifaft64(  int(F_gpu.gpudata), d, delta, segment, axes, normFactor )







