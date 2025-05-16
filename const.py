from numpy.typing import NDArray

import numpy as np

VISUAL_MODE = True

import sys

if sys.gettrace() is not None:
    VISUAL_MODE = False

# VISUAL_MODE = False

FLOAT = np.float32
INT = np.int32
DIM = 2

FloatDxD = NDArray[FLOAT]

FloatN = NDArray[FLOAT]
FloatNxD = NDArray[FLOAT]

FloatS = NDArray[FLOAT]
FloatSx1 = NDArray[FLOAT]

FloatSxD = NDArray[FLOAT]
FloatSxDxD = NDArray[FLOAT]

FloatSxSxD = NDArray[FLOAT]
FloatSxSxDxD = NDArray[FLOAT]

IntS = NDArray[INT]
ExIntS = list[NDArray[INT]]

ExFloatS = list[NDArray[FLOAT]]
ExFloatSxD = list[NDArray[FLOAT]]
ExFloatSxSxD = list[NDArray[FLOAT]]
ExFloatSxSxDxD = list[NDArray[FLOAT]]

FloatT = NDArray[FLOAT]

