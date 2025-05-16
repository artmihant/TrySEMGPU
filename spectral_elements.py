from typing import Any, Generator, Iterator
import numpy as np
from const import *

from numpy.polynomial.legendre import legroots, legder, legval
from scipy.interpolate import lagrange
from itertools import product


def compute_gll_points_and_weights(n_deg: int) -> tuple[FloatS, FloatS]:

    gll_nodes:FloatS = np.array([-1] + [0]*(n_deg - 1) + [1], FLOAT)

    gll_nodes[1:-1] += legroots(legder([0]*n_deg + [1]))

    gll_weights:FloatS = 2/(n_deg *(n_deg + 1)*legval(gll_nodes, [0]*n_deg +[1])**2)

    return gll_nodes, gll_weights


def compute_cube_1d_xinodes(gll_points: FloatS) -> FloatS:
    xi_nodes_d1 = gll_points
    return xi_nodes_d1


def compute_cube_2d_xinodes(gll_points: FloatS) -> FloatS:
    n_gll_nodes = len(gll_points)
    
    xi_nodes_d2 = np.zeros((n_gll_nodes**2, 2), dtype=FLOAT)
    
    for i,j in product(range(n_gll_nodes), range(n_gll_nodes)):
        xi_nodes_d2[i + j*n_gll_nodes] = [gll_points[i],gll_points[j]]

    return xi_nodes_d2


def compute_cube_1d_weights(gll_weights: FloatS) -> FloatS:
    weights_d1 = gll_weights
    return weights_d1 


def compute_cube_2d_weights(gll_weights: FloatS) -> FloatS:
    n_gll_nodes = len(gll_weights)
    
    weights_d2 = np.zeros((n_gll_nodes**2), dtype=FLOAT)
    
    for i,j in product(range(n_gll_nodes), range(n_gll_nodes)):
        weights_d2[i + j*n_gll_nodes] = gll_weights[i]*gll_weights[j]

    return weights_d2


def compute_cube_2d_nabla_shape(gll_nodes: FloatS) -> FloatSxSxD:

    n_gll_nodes = len(gll_nodes)

    nabla_shape_1d = np.zeros((n_gll_nodes**2, n_gll_nodes**2, 2), dtype=FLOAT)

    for i1, j1 in product(range(n_gll_nodes), range(n_gll_nodes)):
        index1 = i1 + j1*n_gll_nodes

        values = np.zeros_like(gll_nodes, dtype=FLOAT)
        values[i1] = 1
        ipoly = lagrange(gll_nodes, values)
        dipoly = np.polyder(ipoly)

        values = np.zeros_like(gll_nodes, dtype=FLOAT)
        values[j1] = 1
        jpoly = lagrange(gll_nodes, values)
        djpoly = np.polyder(jpoly)

        for i2, j2 in product(range(n_gll_nodes), range(n_gll_nodes)):
            index2 = i2 + j2*n_gll_nodes

            if j1 == j2:
                nabla_shape_1d[index2, index1, 0] = np.polyval(dipoly, gll_nodes[i2])
            if i1 == i2:
                nabla_shape_1d[index2, index1, 1] = np.polyval(djpoly, gll_nodes[j2])

    return nabla_shape_1d


def compute_cube_1d_nabla_shape(gll_nodes: FloatS) -> FloatSxSxD:

    n_gll_nodes = len(gll_nodes)

    nabla_shape_2d = np.zeros((n_gll_nodes, n_gll_nodes, 1), dtype=FLOAT)

    for i1 in product(range(n_gll_nodes)):
        index1 = i1

        values = np.zeros_like(gll_nodes, dtype=FLOAT)
        values[i1] = 1
        ipoly = lagrange(gll_nodes, values)
        dipoly = np.polyder(ipoly)
        
        for i2 in product(range(n_gll_nodes)):
            index2 = i2

            nabla_shape_2d[index2, index1, 0] = np.polyval(dipoly, gll_nodes[i2])
        pass

    return nabla_shape_2d



class SpectralElementType:

    weights: FloatS
    dim: int
    deg: int
    family: str
    nabla_shape: FloatSxSxD
    xi_nodes: FloatS

    def __init__(self, family: str, dim: int, deg: int):

        self.dim = dim
        self.deg = deg

        if family == 'cube':

            gll_nodes, gll_weights = compute_gll_points_and_weights(deg)

            if dim == 1:
                self.xi_nodes = compute_cube_1d_xinodes(gll_nodes)
                self.weights = compute_cube_1d_weights(gll_weights)
                self.nabla_shape = compute_cube_1d_nabla_shape(gll_nodes)

            elif dim == 2:
                self.xi_nodes = compute_cube_2d_xinodes(gll_nodes)
                self.weights = compute_cube_2d_weights(gll_weights)
                self.nabla_shape = compute_cube_2d_nabla_shape(gll_nodes)

            else:
                raise Exception(f'{dim}D not supported!')
        else:
            raise Exception(f'{family} not supported!')

    def __len__(self):
        return self.xi_nodes.shape[0]

    @property
    def shape(self):
        return self.xi_nodes.shape


def compute_yacobians(coord: FloatSxD, nabla_shape: FloatSxSxD)-> FloatS:
    return np.linalg.det(compute_yacobi_ms(coord, nabla_shape))

def compute_yacobi_ms(coord: FloatSxD, nabla_shape: FloatSxSxD) -> FloatSxDxD:
    return coord.transpose()@nabla_shape

def compute_antiyacobi_ms(coord: FloatSxD, nabla_shape: FloatSxSxD) -> FloatSxDxD:
    return np.linalg.inv(compute_yacobi_ms(coord,nabla_shape))


class SpectralElement:

    element_type: SpectralElementType
    global_nodes: FloatSxD
    nids: IntS
    k_matrix: FloatSxSxDxD
    weights: FloatSx1

    def __init__(self, global_nodes: FloatNxD, nids: IntS, element_type: SpectralElementType):
        assert nids.shape[0] == element_type.xi_nodes.shape[0]

        self.global_nodes = global_nodes
        self.nids = nids
        self.element_type = element_type

        yacobians = compute_yacobians(self.nodes, element_type.nabla_shape)

        self.weights = (element_type.weights*yacobians).reshape(-1,1)

        n_nodes = len(self)

        self.k_matrix = np.zeros((n_nodes, n_nodes, self.dim, self.dim), dtype=FLOAT)

    @property
    def nodes(self) -> FloatSxD:
        return self.global_nodes[self.nids]

    def __len__(self):
        return self.nids.shape[0]
    
    @property
    def nodes_count(self):
        return self.nodes.shape[0]

    def shape(self):
        return self.nodes.shape

    @property
    def dim(self):
        return self.element_type.dim
    
    @property
    def deg(self):
        return self.element_type.deg
    
    def array(self, dim, dtype=FLOAT):
        return np.zeros((self.nodes_count,dim), dtype=dtype)

    def nabla_shape(self):
        antiyacobi_ms = compute_antiyacobi_ms(self.nodes, self.element_type.nabla_shape)
        return self.element_type.nabla_shape@antiyacobi_ms

    def compute_k_matrix(self, E:float, nu: float):
        
        # Плоская деформация и 3D
        hooke_a = E*(1-nu)/(1+nu)/(1-2*nu)
        hooke_b = E*nu/(1+nu)/(1-2*nu)
        hooke_c = E/2/(1+nu)

        # Плоское напряжение
        # a = E/(1-nu**2)
        # b = E*nu/(1-nu**2)
        # c = E/2/(1+nu)

        if self.dim == 1:
            self.k_matrix[:,:] = np.eye(self.dim, dtype=FLOAT)*E

        elif self.dim == 2:

            n_nodes = len(self)

            nabla_shape = self.nabla_shape()

            nabla_matrix_b = np.zeros_like(self.k_matrix)
            nabla_matrix_b[:,:,0,0] = nabla_shape[:,:,0]
            nabla_matrix_b[:,:,0,1] = nabla_shape[:,:,0]
            nabla_matrix_b[:,:,1,0] = nabla_shape[:,:,1]
            nabla_matrix_b[:,:,1,1] = nabla_shape[:,:,1]

            nabla_matrix_r = np.zeros_like(self.k_matrix)
            nabla_matrix_r[:,:,0,0] = nabla_shape[:,:,0]
            nabla_matrix_r[:,:,0,1] = nabla_shape[:,:,1]
            nabla_matrix_r[:,:,1,0] = nabla_shape[:,:,0]
            nabla_matrix_r[:,:,1,1] = nabla_shape[:,:,1]

            nabla_matrix_t = np.zeros_like(self.k_matrix)
            nabla_matrix_t[:,:,0,0] = nabla_shape[:,:,1]
            nabla_matrix_t[:,:,0,1] = nabla_shape[:,:,1]
            nabla_matrix_t[:,:,1,0] = nabla_shape[:,:,0]
            nabla_matrix_t[:,:,1,1] = nabla_shape[:,:,0]

            nabla_matrix_l = np.zeros_like(self.k_matrix)
            nabla_matrix_l[:,:,0,0] = nabla_shape[:,:,1]
            nabla_matrix_l[:,:,0,1] = nabla_shape[:,:,0]
            nabla_matrix_l[:,:,1,0] = nabla_shape[:,:,1]
            nabla_matrix_l[:,:,1,1] = nabla_shape[:,:,0]

            for mu in range(n_nodes):
                for la in range(n_nodes):
                    
                    k_matrix_block = (self.weights.reshape(-1,1,1)*(
                        np.array([
                            [hooke_a, hooke_b],
                            [hooke_b, hooke_a]
                        ]) * nabla_matrix_r[:,la] * nabla_matrix_b[:,mu] + 
                        np.array([
                            [hooke_c, hooke_c],
                            [hooke_c, hooke_c]
                        ]) * nabla_matrix_l[:,la] * nabla_matrix_t[:,mu]
                    )).sum(0)

                    self.k_matrix[la, mu] = k_matrix_block


class SpectralMesh:

    elements: list[SpectralElement]
    nodes: FloatNxD

    def __init__(self, nodes: FloatNxD, elements: list[SpectralElement]):
        self.nodes = nodes
        self.elements = elements

    @property
    def nodes_count(self) -> int:
        return self.nodes.shape[0]
    
    def dim(self):
        return self.nodes.shape[1]
    
    def array(self, dim, dtype=FLOAT):
        return np.zeros((self.nodes_count,dim), dtype=dtype)