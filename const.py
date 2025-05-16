from numpy.typing import NDArray

import numpy as np

VISUAL_MODE = True

import sys

if sys.gettrace() is not None:
    VISUAL_MODE = False

# VISUAL_MODE = False

FLOAT = np.float64
INT = np.int64
DIM = 2

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

