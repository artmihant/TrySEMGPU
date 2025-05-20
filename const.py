from numpy.typing import NDArray
from numba.cuda.cudadrv.devicearray import DeviceNDArray

import numpy as np

VISUAL_MODE = True

import sys

if sys.gettrace() is not None:
    VISUAL_MODE = False

VISUAL_MODE = False

FLOAT = np.float32
INT = np.int32
DIM = 2
THREADS_COUNT = 128

FloatDxD = NDArray[FLOAT]

FloatN = NDArray[FLOAT]
FloatNx1 = NDArray[FLOAT]
FloatNxD = NDArray[FLOAT]

FloatS = NDArray[FLOAT]
FloatSx1 = NDArray[FLOAT]

FloatSxD = NDArray[FLOAT]
FloatSxDxD = NDArray[FLOAT]

FloatSxSxD = NDArray[FLOAT]
FloatSxSxDxD = NDArray[FLOAT]

IntA = NDArray[INT]
IntS = NDArray[INT]
IntN = NDArray[INT]
IntE = NDArray[INT]
IntEN = NDArray[INT]

FloatA = NDArray[FLOAT]

FloatE = NDArray[FLOAT]
FloatT = NDArray[FLOAT]
FloatENx1 = NDArray[FLOAT]

DIntA = DeviceNDArray
DIntE = DeviceNDArray
DFloatEx1 = DeviceNDArray
DIntEN = DeviceNDArray
DFloatS = DeviceNDArray
DFloatSxSxD = DeviceNDArray
DFloatNxD = DeviceNDArray
DFloatNx1 = DeviceNDArray
DFloatENx1 = DeviceNDArray
DFloatEN = DeviceNDArray
DFloatENxDxD = DeviceNDArray