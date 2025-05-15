# VISUAL_MODE = True
VISUAL_MODE = False


import time
import numpy as np

import numpy as np


from typing import Annotated, Literal
from numpy.typing import NDArray
import numpy as np

from scipy.interpolate import lagrange
import numpy as np
from numpy.polynomial.legendre import legroots, legder, legval
from itertools import product

from numba import njit, prange, types


FloatC = Annotated[NDArray[np.float64], Literal[('S')]]
FloatS = Annotated[NDArray[np.float64], Literal[('S')]]

FloatN = Annotated[NDArray[np.float64], Literal[('N')]]
FloatNx2 = Annotated[NDArray[np.float64], Literal[('N', 2)]]
FloatNx3 = Annotated[NDArray[np.float64], Literal[('N', 4)]]

FloatSx2 = Annotated[NDArray[np.float64], Literal[('S', 2)]]
FloatSx2x2 = Annotated[NDArray[np.float64], Literal[('S', 2, 2)]]
FloatSxSx2 = Annotated[NDArray[np.float64], Literal[('S', 'S', 2)]]
FloatСxСx1 = Annotated[NDArray[np.float64], Literal[('S', 'S', 1)]]

FloatSx3 = Annotated[NDArray[np.float64], Literal[('S', 6)]]

Float2x2 = Annotated[NDArray[np.float64], Literal[(2, 2)]]
Float3x3 = Annotated[NDArray[np.float64], Literal[(3, 3)]]

IntExS = Annotated[NDArray[np.int64], Literal[('E', 'S')]]

IntSxS = Annotated[NDArray[np.int64], Literal[('S', 'S')]]


def compute_gll(S1: int) -> tuple[FloatC, FloatC]:

    GLL_points:FloatC = np.array([-1] + [0]*(S1-1) + [1], np.float64)

    GLL_points[1:-1] += legroots(legder([0]*S1+[1])) # type: ignore

    GLL_weights:FloatC = 2/(S1*(S1+1)*legval(GLL_points, [0]*S1+[1])**2) # type: ignore

    return GLL_points, GLL_weights


def riker(f:float, A:float, t0:float, t1:float, dt:float):
    return \
        (lambda t: (t, 
            (lambda p: A*(1+2*p)*np.exp(p))
            (-(np.pi*(f*(t-t0)-1))**2))
        )(np.arange(t0, t1, dt))



def compute_NablaShape2(SpectralPoints1: FloatC) -> FloatSxSx2:

    length = len(SpectralPoints1)

    NablaShape = np.zeros(((length)**2, (length)**2, 2))

    for i1, j1 in product(range(length), range(length)):
        index1 = i1 + j1*length

        values = np.zeros_like(SpectralPoints1, dtype=float)
        values[i1] = 1
        ipoly = lagrange(SpectralPoints1, values)
        dipoly = np.polyder(ipoly)

        values = np.zeros_like(SpectralPoints1, dtype=float)
        values[j1] = 1
        jpoly = lagrange(SpectralPoints1, values)
        djpoly = np.polyder(jpoly)

        for i2, j2 in product(range(length), range(length)):
            index2 = i2 + j2*length

            if j1 == j2:
                NablaShape[index2, index1, 0] = np.polyval(dipoly, SpectralPoints1[i2])
            if i1 == i2:
                NablaShape[index2, index1, 1] = np.polyval(djpoly, SpectralPoints1[j2])

    return NablaShape



def compute_NablaShape1(SpectralPoints1: FloatC) -> FloatСxСx1:

    length = len(SpectralPoints1)

    NablaShape = np.zeros((length, length, 1))

    for i1 in product(range(length)):
        index1 = i1

        values = np.zeros_like(SpectralPoints1, dtype=float)
        values[i1] = 1
        ipoly = lagrange(SpectralPoints1, values)
        dipoly = np.polyder(ipoly)
        
        for i2 in product(range(length)):
            index2 = i2

            NablaShape[index2, index1, 0] = np.polyval(dipoly, SpectralPoints1[i2])
        pass

    return NablaShape



def compute_SpectralWeights2(SpectralWeights1: FloatC) -> FloatS:
    length = len(SpectralWeights1)
    
    SpectralWeights = np.zeros((length**2, 1))
    
    for i,j in product(range(length), range(length)):
        SpectralWeights[i + j*length] = SpectralWeights1[i]*SpectralWeights1[j]

    return SpectralWeights


def cofactor_matrix_2x2(m: Float2x2) -> Float2x2:
    return np.array([
        [m[1,1], -m[0,1]],
        [-m[1,0], m[0,0]]
    ])

# @jit
def compute_Yacobi(coord: FloatSx2, nabla_shape: FloatSxSx2) -> FloatSx2x2:
    return coord.transpose()@nabla_shape # type: ignore

# @njit(types.Array(types.float64, 3, 'C')(types.Array(types.float64, 2, 'C'), types.Array(types.float64, 3, 'C')))
# def compute_Yacobi(coord: FloatSx2, nabla_shape: FloatSxSx2) -> FloatSx2x2:
#     ElementSize = coord.shape[0]
#     yacobi = np.zeros((ElementSize, 2, 2))

#     tcoord = coord.transpose()

#     for i in range(ElementSize):
#         yacobi[i] = tcoord@nabla_shape[i]
#     return yacobi # type: ignore


def compute_Coyacobi(coord: FloatSx2, nabla_shape: FloatSxSx2) -> FloatSx2x2:
    yacobi = compute_Yacobi(coord, nabla_shape)
    for i in range(len(yacobi)):
        yacobi[i] = cofactor_matrix_2x2(yacobi[i])

    return yacobi


def compute_Yacobian(coord: FloatSx2, nabla_shape: FloatSxSx2)-> FloatS:
    return np.linalg.det(compute_Yacobi(coord, nabla_shape)).reshape(-1,1)



def generate_spectral_box(GridSize, ElementSize, StartPoint, SpectralPoints1) -> tuple[FloatSx2, IntExS]:

    SpectralDeg = len(SpectralPoints1) - 1

    x_nodes_count = (GridSize[0]*SpectralDeg + 1)
    y_nodes_count = (GridSize[1]*SpectralDeg + 1)

    nodes_count = x_nodes_count*y_nodes_count

    spectral_size2 = (SpectralDeg+1)**2

    Coord = np.zeros((nodes_count, 2), np.float64)

    Elems = np.zeros((GridSize[0]*GridSize[1], spectral_size2), np.int64)

    for ey, ex in product(range(GridSize[1]), range(GridSize[0])):

        for sy, sx in product(range(SpectralDeg+1), range(SpectralDeg+1)):

            nx = sx + SpectralDeg*ex 
            ny = sy + SpectralDeg*ey

            node_index = nx + x_nodes_count*ny

            Coord[node_index] = [
                StartPoint[0] + (ex + (SpectralPoints1[sx]+1)*0.5 ) * ElementSize,
                StartPoint[1] + (ey + (SpectralPoints1[sy]+1)*0.5 ) * ElementSize
            ]

            element_index  = ex + GridSize[0]*ey
            spectral_index = sx + (SpectralDeg+1)*sy

            Elems[element_index][spectral_index] = node_index

            pass

    return Coord, Elems


def ForceSigma(
        Sigma: FloatSx3,
        NablaOp
    ):
    
    return (NablaOp@Sigma[:, [
        [0, 2],
        [2, 1],
    ]]).sum(0)


def ForceGrav(
        SpectralWeights: FloatS,
        GravCoef: FloatSx2,
        Yacobian: FloatS
    ) -> FloatSx2:
    return SpectralWeights*GravCoef*Yacobian # type: ignore


def SigmaDerivative(
        CoYacobi: FloatSx2x2, 
        SpectralWeights: FloatS,
        NablaShape: FloatSxSx2, 
        Velocity: FloatSx2, 
        Hooke: Float3x3, 
        Volumes: FloatS,
    ):
    SpertralSize = len(SpectralWeights)

    GradVelocity: FloatSx2x2 = np.zeros((SpertralSize, 2, 2))

    NablaOp = SpectralWeights*NablaShape@CoYacobi
    
    NablaOp = NablaOp.reshape(*NablaOp.shape, 1)
    Velocity = Velocity.reshape(-1, 1, 2)

    for lam in range(SpertralSize): 
        GradVelocity[lam] += (NablaOp[lam]@Velocity).sum(axis=0)

    Velocity = Velocity.reshape(-1, 2)

    Epsilon = ((GradVelocity + GradVelocity.transpose(0,2,1)) * 0.5) # type: ignore

    Epsilon: FloatSx3 = Epsilon[:,(0,1,0),(0,1,1)]

    Sigma = (Epsilon@Hooke)/Volumes
    
    return Sigma



def CalculationProcess(
        tau: float,
        step: float,
        PulseNodeIndex, 
        riker_wavelet,
        Velocity: FloatNx2, 
        Coord: FloatNx2, 
        Sigma: FloatNx2, 
        Elems: IntExS,
        Mass: FloatN,
        Volumes: FloatN,
        Hooke: Float3x3,
        NablaShape2: FloatSxSx2, 
        SpectralWeights2: FloatS,
    ):

    for elem in Elems:
        CoYacobi = compute_Coyacobi(Coord[elem], NablaShape2)

        NablaOp = SpectralWeights2*NablaShape2@CoYacobi

        Velocity[elem] -= tau*ForceSigma(Sigma[elem], NablaOp)/Mass[elem]

    # if step < riker_wavelet.shape[0]:
    #     Velocity[PulseNodeIndex][1] += tau*riker_wavelet[step]/Mass[PulseNodeIndex][0]

    Coord += tau*Velocity*0.5

    Volumes.fill(0)

    for elem in Elems:
        Yacobian = compute_Yacobian(Coord[elem], NablaShape2)
        Volumes[elem] += SpectralWeights2*Yacobian

    for elem in Elems:

        CoYacobi = compute_Coyacobi(Coord[elem], NablaShape2)
        Yacobian = compute_Yacobian(Coord[elem], NablaShape2)
        
        Sigma[elem] += tau*SigmaDerivative(CoYacobi, SpectralWeights2, NablaShape2, Velocity[elem], Hooke, Volumes[elem])

    Coord += tau*Velocity*0.5


def compute_Mass(StaticDensity: float, Coord: FloatNx2, Elems: IntExS, NablaShape2:FloatSxSx2, SpectralWeights2: FloatS) -> FloatN:

    nodes_count = Coord.shape[0]
    Density = np.full((nodes_count, 1), StaticDensity, dtype=np.float64)
    Mass = np.zeros_like(Density)
    
    for elem in Elems:
        Yacobian = compute_Yacobian(Coord[elem], NablaShape2)
        adding = SpectralWeights2*Density[elem]*Yacobian
        Mass[elem] += adding

    return Mass


def compute_Volume(Coord: FloatNx2, Elems: IntExS, NablaShape2:FloatSxSx2, SpectralWeights2: FloatS):

    nodes_count = Coord.shape[0]
    Volume = np.zeros((nodes_count, 1))

    for elem in Elems:
        Yacobian = compute_Yacobian(Coord[elem], NablaShape2)
        Volume[elem] += SpectralWeights2*Yacobian

    return Volume

def compute_Hooke(E: float, nu: float) -> Float3x3:

    a = (1-nu)/(1-2*nu)
    b = nu/(1-2*nu)
    c = 1

    return np.array([
        [a, b, 0],
        [b, a, 0],
        [0, 0, c]
    ])*E/(1+nu) # type: ignore


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def main():

    SpectralDeg = 7

    SpaceSize = [20, 10]

    ElementSize = 1
    TotalSimulationTime = 1
    TotalSteps = 400

    GridSize = [int(SpaceSize[0]//ElementSize), int(SpaceSize[1]//ElementSize)]

    StartPoint = [0, 0]
    PulsePoint = [GridSize[0]//2, GridSize[1]]

    StaticYoung = 100000
    StaticDensity = 1000
    StaticPoisson = 0.25

    StaticVp = (StaticYoung/StaticDensity*(1-StaticPoisson)/(1+StaticPoisson)/(1-2*StaticPoisson))**0.5
    StaticVs = (StaticYoung/StaticDensity/2/(1+StaticPoisson))**0.5

    RikerPulseAmplitude = 100
    RikerPulseFrequency = 10

    tau = TotalSimulationTime/TotalSteps

    SpectralPoints1, SpectralWeights1 = compute_gll(SpectralDeg)
    
    NablaShape2 = compute_NablaShape2(SpectralPoints1)
    NablaShape1 = compute_NablaShape1(SpectralPoints1)

    SpectralWeights2 = compute_SpectralWeights2(SpectralWeights1)

    Coord, Elems = generate_spectral_box(GridSize, ElementSize, StartPoint, SpectralPoints1)

    Nodes_count = Coord.shape[0]

    Mass = compute_Mass(StaticDensity, Coord, Elems, NablaShape2, SpectralWeights2)

    Velocity = np.zeros((Nodes_count, 2))
    Sigma = np.zeros((Nodes_count, 3))

    _, riker_wavelet = riker(RikerPulseFrequency, RikerPulseAmplitude, 0, TotalSimulationTime, tau)

    # GravCoef = np.zeros((Nodes_count, 2))
    Volumes = np.zeros((Nodes_count, 1))

    x_nodes_count = (GridSize[0]*SpectralDeg + 1)
    y_nodes_count = (GridSize[1]*SpectralDeg + 1)

    PulseNodeIndex = SpectralDeg*(PulsePoint[0] + x_nodes_count*PulsePoint[1])

    Hooke = compute_Hooke(StaticYoung, StaticPoisson)


    if VISUAL_MODE:

        VisualField = np.zeros((y_nodes_count//SpectralDeg, x_nodes_count//SpectralDeg))

        fig, ax = plt.subplots()
        im = ax.imshow(VisualField, cmap='viridis', vmin=0, vmax=0.005)


        def update(step):

            CalculationProcess(
                tau,
                step,
                PulseNodeIndex, 
                riker_wavelet, 
                Velocity, 
                Coord, 
                Sigma, 
                Elems,
                Mass,
                Volumes,
                Hooke,
                NablaShape2, 
                SpectralWeights2,
            )

            KineticEnergy = (Mass.ravel()*(Velocity[:, 0]**2 + Velocity[:, 1]**2)).reshape(y_nodes_count, x_nodes_count)[::-1]

            PotencialEnergy = (Sigma[:, 0]**2 + Sigma[:, 1]**2 + 2*Sigma[:, 2]**2).reshape(y_nodes_count, x_nodes_count)[::-1]/StaticYoung

            VisualField = (10*KineticEnergy + PotencialEnergy)[::2,::2]

            a2, b2 = np.sum(PotencialEnergy), np.sum(KineticEnergy)

            print(step, a2, b2, np.max(VisualField), np.max(Velocity))

            im.set_data(VisualField)  # обновляем данные среза
            return [im]

        ani = FuncAnimation(fig, update, interval=10)
        plt.show()

    else:
        step_duration = time.time()

        for i in range(400):
            print(i)
            CalculationProcess(
                tau,
                100,
                PulseNodeIndex, 
                riker_wavelet, 
                Velocity, 
                Coord, 
                Sigma, 
                Elems,
                Mass,
                Volumes,
                Hooke,
                NablaShape2, 
                SpectralWeights2,
            )

        print('Done:', time.time() - step_duration)


if __name__ == '__main__':

    main()

