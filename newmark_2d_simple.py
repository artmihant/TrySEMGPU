
VISUAL_MODE = True

import sys

if sys.gettrace() is not None:
    VISUAL_MODE = False

# VISUAL_MODE = False
import time

import numpy as np

from typing import Any
import numpy as np
from numpy.typing import NDArray

from scipy.interpolate import lagrange
from numpy.polynomial.legendre import legroots, legder, legval
from itertools import product

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

IntExS2 = NDArray[np.int64]

Float2x2 = NDArray[np.float64]

FloatS1 = NDArray[np.float64]
FloatS1xS1x1 = NDArray[np.float64]

FloatS2 = NDArray[np.float64]

FloatS2x1 = NDArray[np.float64]
FloatS2x2 = NDArray[np.float64]
FloatS2x2x2 = NDArray[np.float64]

FloatS2xS2x2 = NDArray[np.float64]
FloatS2xS2x2x2 = NDArray[np.float64]

FloatN = NDArray[np.float64]
FloatNx1 = NDArray[np.float64]
FloatNx2 = NDArray[np.float64]

FloatExS2 = NDArray[np.float64]
FloatExS2x1 = NDArray[np.float64]
FloatExS2xS2x2 = NDArray[np.float64]
FloatExS2xS2x2x2 = NDArray[np.float64]

FloatT = NDArray[np.float64]


def compute_gll_points_and_weights(n_deg: int) -> tuple[FloatS1, FloatS1]:

    gll_points:FloatS1 = np.array([-1] + [0]*(n_deg - 1) + [1], np.float64)

    gll_points[1:-1] += legroots(legder([0]*n_deg + [1]))

    gll_weights:FloatS1 = 2/(n_deg *(n_deg + 1)*legval(gll_points, [0]*n_deg +[1])**2)

    return gll_points, gll_weights


def compute_riker_pulse(f:float, A:float, t0:float, t1:float, dt:float) -> FloatT:
    return \
        (lambda t: 
            (lambda p: A*(1+2*p)*np.exp(p))
            (-(np.pi*(f*(t-t0)-1))**2)
        )(np.arange(t0, t1, dt))


def compute_nabla_shape_d2(xi_nodes_d1: FloatS1) -> FloatS2xS2x2:

    n_nodes_1d = len(xi_nodes_d1)

    nabla_shape_d1 = np.zeros((n_nodes_1d**2, n_nodes_1d**2, 2))

    for i1, j1 in product(range(n_nodes_1d), range(n_nodes_1d)):
        index1 = i1 + j1*n_nodes_1d

        values = np.zeros_like(xi_nodes_d1, dtype=float)
        values[i1] = 1
        ipoly = lagrange(xi_nodes_d1, values)
        dipoly = np.polyder(ipoly)

        values = np.zeros_like(xi_nodes_d1, dtype=float)
        values[j1] = 1
        jpoly = lagrange(xi_nodes_d1, values)
        djpoly = np.polyder(jpoly)

        for i2, j2 in product(range(n_nodes_1d), range(n_nodes_1d)):
            index2 = i2 + j2*n_nodes_1d

            if j1 == j2:
                nabla_shape_d1[index2, index1, 0] = np.polyval(dipoly, xi_nodes_d1[i2])
            if i1 == i2:
                nabla_shape_d1[index2, index1, 1] = np.polyval(djpoly, xi_nodes_d1[j2])

    return nabla_shape_d1


def compute_nabla_shape_d1(xi_nodes_d1: FloatS1) -> FloatS1xS1x1:

    n_nodes_1d = len(xi_nodes_d1)

    nabla_shape_d2 = np.zeros((n_nodes_1d, n_nodes_1d, 1))

    for i1 in product(range(n_nodes_1d)):
        index1 = i1

        values = np.zeros_like(xi_nodes_d1, dtype=float)
        values[i1] = 1
        ipoly = lagrange(xi_nodes_d1, values)
        dipoly = np.polyder(ipoly)
        
        for i2 in product(range(n_nodes_1d)):
            index2 = i2

            nabla_shape_d2[index2, index1, 0] = np.polyval(dipoly, xi_nodes_d1[i2])
        pass

    return nabla_shape_d2


def compute_weights_d2(weights_d1: FloatS1) -> FloatS2:
    n_nodes_1d = len(weights_d1)
    
    weights_d2 = np.zeros((n_nodes_1d**2))
    
    for i,j in product(range(n_nodes_1d), range(n_nodes_1d)):
        weights_d2[i + j*n_nodes_1d] = weights_d1[i]*weights_d1[j]

    return weights_d2


def cofactor_matrix_2x2(m: Float2x2) -> Float2x2:
    return np.array([
        [m[1,1], -m[0,1]],
        [-m[1,0], m[0,0]]
    ])


def compute_yacobi_ms(coord: FloatS2x2, nabla_shape: FloatS2xS2x2) -> FloatS2x2x2:
    return coord.transpose()@nabla_shape


def compute_antiyacobi_ms(coord: FloatS2x2, nabla_shape: FloatS2xS2x2) -> FloatS2x2x2:
    return np.linalg.inv(compute_yacobi_ms(coord,nabla_shape))


def compute_coyacobi_ms(coord: FloatS2x2, nabla_shape: FloatS2xS2x2) -> FloatS2x2x2:
    yacobi = compute_yacobi_ms(coord, nabla_shape)
    for i in range(len(yacobi)):
        yacobi[i] = cofactor_matrix_2x2(yacobi[i])

    return yacobi


def compute_yacobians(coord: FloatS2x2, nabla_shape: FloatS2xS2x2)-> FloatS2:
    return np.linalg.det(compute_yacobi_ms(coord, nabla_shape))


def generate_spectral_mesh_box(grid_size: tuple[int, int], single_element_size: float, start_point: tuple[float, float], xi_nodes_d1:FloatS1) -> tuple[FloatS2x2, IntExS2]:

    n_nodes_1d = len(xi_nodes_d1)
    n_deg = n_nodes_1d - 1

    x_nodes_count = (grid_size[0]*n_deg + 1)
    y_nodes_count = (grid_size[1]*n_deg + 1)

    nodes_count = x_nodes_count*y_nodes_count

    n_nodes_2d = n_nodes_1d**2

    global_coords = np.zeros((nodes_count, 2), np.float64)

    global_elems = np.zeros((grid_size[0]*grid_size[1], n_nodes_2d), np.int64)

    for ey, ex in product(range(grid_size[1]), range(grid_size[0])):

        for sy, sx in product(range(n_nodes_1d), range(n_nodes_1d)):

            nx = sx + n_deg*ex 
            ny = sy + n_deg*ey

            node_index = nx + x_nodes_count*ny

            global_coords[node_index] = [
                start_point[0] + (ex + (xi_nodes_d1[sx]+1)*0.5 ) * single_element_size,
                start_point[1] + (ey + (xi_nodes_d1[sy]+1)*0.5 ) * single_element_size
            ]

            element_index  = ex + grid_size[0]*ey
            spectral_index = sx + n_nodes_1d*sy

            global_elems[element_index][spectral_index] = node_index

            pass

    return global_coords, global_elems



def compute_global_K_matrixes(global_elems, global_nodes, nabla_shape_d2, weights_d2, hooke_a, hooke_b, hooke_c):

    global_K_matrixes = []

    for eid, elem in enumerate(global_elems):

        n_nodes_2d = elem.shape[0]

        yacobians = compute_yacobians(global_nodes[elem], nabla_shape_d2)
        antiyacobi_ms = compute_antiyacobi_ms(global_nodes[elem], nabla_shape_d2)

        local_nablas = nabla_shape_d2@antiyacobi_ms
        local_weights = weights_d2*yacobians

        local_K_matrixes = np.zeros((n_nodes_2d, n_nodes_2d, 2, 2))

        NablaMatrixB = np.zeros((n_nodes_2d,n_nodes_2d,2,2))
        NablaMatrixB[:,:,0,0] = local_nablas[:,:,0]
        NablaMatrixB[:,:,0,1] = local_nablas[:,:,0]
        NablaMatrixB[:,:,1,0] = local_nablas[:,:,1]
        NablaMatrixB[:,:,1,1] = local_nablas[:,:,1]

        NablaMatrixR = np.zeros((n_nodes_2d,n_nodes_2d,2,2))
        NablaMatrixR[:,:,0,0] = local_nablas[:,:,0]
        NablaMatrixR[:,:,0,1] = local_nablas[:,:,1]
        NablaMatrixR[:,:,1,0] = local_nablas[:,:,0]
        NablaMatrixR[:,:,1,1] = local_nablas[:,:,1]

        NablaMatrixT = np.zeros((n_nodes_2d,n_nodes_2d,2,2))
        NablaMatrixT[:,:,0,0] = local_nablas[:,:,1]
        NablaMatrixT[:,:,0,1] = local_nablas[:,:,1]
        NablaMatrixT[:,:,1,0] = local_nablas[:,:,0]
        NablaMatrixT[:,:,1,1] = local_nablas[:,:,0]

        NablaMatrixL = np.zeros((n_nodes_2d,n_nodes_2d,2,2))
        NablaMatrixL[:,:,0,0] = local_nablas[:,:,1]
        NablaMatrixL[:,:,0,1] = local_nablas[:,:,0]
        NablaMatrixL[:,:,1,0] = local_nablas[:,:,1]
        NablaMatrixL[:,:,1,1] = local_nablas[:,:,0]

        for mu in range(n_nodes_2d):
            for la in range(n_nodes_2d):
                
                KMatrix = (local_weights.reshape(-1,1,1)*(
                    np.array([
                        [hooke_a, hooke_b],
                        [hooke_b, hooke_a]
                    ]) * NablaMatrixR[:,la] * NablaMatrixB[:,mu] + 
                    np.array([
                        [hooke_c, hooke_c],
                        [hooke_c, hooke_c]
                    ]) * NablaMatrixL[:,la] * NablaMatrixT[:,mu]
                )).sum(0)

                local_K_matrixes[la, mu] = KMatrix

        global_K_matrixes.append(local_K_matrixes)

    return global_K_matrixes



def simulation_loop(
        tau: float,
        step: int,
        riker_pulse_node_id: int, 
        riker_pulse: FloatT,
        global_displacement: FloatNx2, 
        global_velocity: FloatNx2, 
        global_acceleration: FloatNx2, 
        global_elems: IntExS2,
        global_mass: FloatNx1,
        global_K_matrixes: FloatExS2xS2x2x2
    ) -> None:

    """ получаем eid элемента, с которым собираемся работать """

    for eid, elem in enumerate(global_elems):
        n_nodes_2d = elem.shape[0]
    
        elem_K_matrix: FloatS2xS2x2x2 = global_K_matrixes[eid]
        elem_displacement = global_displacement[elem]

        elem_inner_force = np.zeros((n_nodes_2d, 2))

        for la in range(n_nodes_2d):
            elem_inner_force[la] += (elem_K_matrix[la] @ (elem_displacement.reshape(n_nodes_2d, 2, 1))).sum(0).reshape(2)

        global_acceleration[elem] -= elem_inner_force / global_mass[elem]


    global_velocity += tau*global_acceleration
    global_displacement += tau*global_velocity

    global_acceleration[:] = 0

    if step < riker_pulse.shape[0]:
        global_acceleration[riker_pulse_node_id][1] += riker_pulse[step]



def compute_global_mass(constant_density: float, global_coords: FloatNx2, global_elems: IntExS2, nabla_shape_d2:FloatS2xS2x2, weights_d2: FloatS2) -> FloatNx1:

    global_n_nodes = global_coords.shape[0]
    
    density = np.full((global_n_nodes, 1), constant_density, dtype=np.float64)

    global_mass = np.zeros_like(density)
    
    for elem in global_elems:
        yacobians = compute_yacobians(global_coords[elem], nabla_shape_d2)
        global_mass[elem] += weights_d2.reshape((-1,1))*density[elem]*yacobians.reshape((-1,1))

    return global_mass


def compute_global_volumes(global_coords: FloatNx2, global_elems: IntExS2, nabla_shape_d2:FloatS2xS2x2, weights_d2: FloatS2) -> FloatNx1:

    nodes_count = global_coords.shape[0]
    Volume = np.zeros((nodes_count, 1))

    for elem in global_elems:
        Yacobian = compute_yacobians(global_coords[elem], nabla_shape_d2)
        Volume[elem] += weights_d2*Yacobian

    return Volume

def compute_hooke_constants(E: float, nu: float) -> tuple[float,float,float]:

    # Плоская деформация
    a = E*(1-nu)/(1+nu)/(1-2*nu)
    b = E*nu/(1+nu)/(1-2*nu)
    c = E/2/(1+nu)

    # Плоское напряжение
    # a = E/(1-nu**2)
    # b = E*nu/(1-nu**2)
    # c = E/2/(1+nu)

    return a,b,c



def main() -> None:

    # Порядок спектрального элемента
    n_deg = 7

    # Физический размер пластины
    plate_size = (20, 10)

    # Размер (квадратного) элемента
    single_element_size = 1

    # Сремя симуляции (сек)
    total_simulation_time = 0.5

    # Количество шагов симуляции
    total_steps = 428

    # Шаг по времени
    tau = total_simulation_time/total_steps

    # Размер спектральной сетки
    grid_size = (int(plate_size[0]//single_element_size), int(plate_size[1]//single_element_size))

    # Начальная точка
    start_point = (0, 0)

    # Параметры материала
    constan_young = 100000
    constant_density = 1000
    constan_poisson = 0.25

    # Скорость распространения волны в этом материале
    constant_Vp = (constan_young/constant_density*(1-constan_poisson)/(1+constan_poisson)/(1-2*constan_poisson))**0.5
    constant_Vs = (constan_young/constant_density/2/(1+constan_poisson))**0.5

    # Набор спектральных точек и весов для первого порядка
    xi_nodes_d1, weights_d1 = compute_gll_points_and_weights(n_deg)
    
    # Набор значений градиентов для каждой точки для всех точек внутри спектрального элемента. Для второй и первой размерности
    nabla_shape_d2: FloatS2xS2x2 = compute_nabla_shape_d2(xi_nodes_d1)

    # Спектральные веса для первого порядка. Точки на практике не нужны.
    weights_d2:FloatS2 = compute_weights_d2(weights_d1)

    # Генератор сетки
    global_nodes, global_elems = generate_spectral_mesh_box(grid_size, single_element_size, start_point, xi_nodes_d1)


    # Точка возмущения Рикера
    riker_pulse_point = [grid_size[0]//2, grid_size[1]]
    riker_pulse_amplitude = 100
    riker_pulse_frequency = 5

    # 
    riker_pulse = compute_riker_pulse(riker_pulse_frequency, riker_pulse_amplitude, 0, total_simulation_time, tau)

    x_nodes_count = (grid_size[0]*n_deg + 1)
    y_nodes_count = (grid_size[1]*n_deg + 1)

    # Индекс узла, к которому будет применено возмущение
    pulse_node_id = n_deg*(riker_pulse_point[0] + x_nodes_count*riker_pulse_point[1])
    
    global_n_nodes = global_nodes.shape[0]

    global_mass = compute_global_mass(constant_density, global_nodes, global_elems, nabla_shape_d2, weights_d2)

    global_displacement = np.zeros((global_n_nodes, 2))
    global_velocity = np.zeros((global_n_nodes, 2))
    global_acceleration = np.zeros((global_n_nodes, 2))

    hooke_a, hooke_b, hooke_c = compute_hooke_constants(constan_young, constan_poisson)


    global_K_matrixes = compute_global_K_matrixes(global_elems, global_nodes, nabla_shape_d2, weights_d2, hooke_a, hooke_b, hooke_c)



    if VISUAL_MODE:

        visual_field = np.zeros((y_nodes_count, x_nodes_count))

        fig, ax = plt.subplots()
        im = ax.imshow(visual_field, cmap='viridis', vmin=0, vmax=0.001)
        
        step_duration = time.time()
        
        def update(step: int) -> list[Any]:
            
            simulation_loop(
                tau,
                step,
                pulse_node_id, 
                riker_pulse,
                global_displacement, 
                global_velocity, 
                global_acceleration, 
                global_elems,
                global_mass,
                global_K_matrixes
            )
            visual_field = (global_mass.ravel()*(global_velocity[:, 0]**2 + global_velocity[:, 1]**2)).reshape(y_nodes_count, x_nodes_count)

            print(step, time.time() - step_duration, np.sum(visual_field), np.max(visual_field))

            im.set_data(visual_field)  # обновляем данные среза
            # im.set_clim(vmin=visual_field.min(), vmax=visual_field.max())

            return [im]

        ani = FuncAnimation(fig, update, interval=10)
        plt.show()

    else:
        
        for step in range(total_steps):
            step_duration = time.time()

            simulation_loop(
                tau,
                step,
                pulse_node_id, 
                riker_pulse,
                global_displacement, 
                global_velocity, 
                global_acceleration, 
                global_elems,
                global_mass,
                global_K_matrixes
            )

            print(step, time.time() - step_duration)


if __name__ == '__main__':

    main()

