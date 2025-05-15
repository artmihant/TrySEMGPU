
VISUAL_MODE = True

import sys

if sys.gettrace() is not None:
    VISUAL_MODE = False

# VISUAL_MODE = False


THREADS_COUNT = 256

# VISUAL_MODE = False
import time

import numpy as np
from numba import cuda, float32

from typing import Any
import numpy as np
from numpy.typing import NDArray

from scipy.interpolate import lagrange
from numpy.polynomial.legendre import legroots, legder, legval
from itertools import product
from numba.cuda.cudadrv.devicearray import DeviceNDArray

from numba import njit

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

IntExS2 = NDArray[np.int64]

Float2x2 = NDArray[np.float64]
Float3x3 = NDArray[np.float64]

FloatS1 = NDArray[np.float64]
FloatS1xS1x1 = NDArray[np.float64]

FloatS2 = NDArray[np.float64]

FloatS2x1 = NDArray[np.float64]
FloatS2x2 = NDArray[np.float64]
FloatS2x2x2 = NDArray[np.float64]

FloatS2xS2x2 = NDArray[np.float64]

FloatN = NDArray[np.float64]
FloatNx1 = NDArray[np.float64]
FloatNx2 = NDArray[np.float64]

FloatExS2 = NDArray[np.float64]
FloatExS2x1 = NDArray[np.float64]
FloatExS2xS2x2 = NDArray[np.float64]

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


# @njit(types.Array(types.float64, 3, 'C')(types.Array(types.float64, 2, 'C'), types.Array(types.float64, 3, 'C')))


@cuda.jit
def change_velocity_and_displacement(
        global_n_nodes: int,
        tau: float, 
        step: int, 
        riker_pulse_node_id: int,  
        riker_pulse: FloatT, 
        d_global_displacement_x: FloatN, 
        d_global_displacement_y: FloatN, 
        d_global_velocity_x: FloatN, 
        d_global_velocity_y: FloatN, 
        d_global_acceleration_x: FloatN,
        d_global_acceleration_y: FloatN

    ) -> None:

    nid = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if nid >= global_n_nodes:
        return

    d_global_velocity_x[nid] += tau*d_global_acceleration_x[nid]
    d_global_velocity_y[nid] += tau*d_global_acceleration_y[nid]
    d_global_displacement_x[nid] += tau*d_global_velocity_x[nid]
    d_global_displacement_y[nid] += tau*d_global_velocity_y[nid]

    d_global_acceleration_x[nid] = 0
    d_global_acceleration_y[nid] = 0

    if nid == riker_pulse_node_id and step < riker_pulse.shape[0]:
        d_global_acceleration_y[nid] += riker_pulse[step]


@cuda.jit(device=True)
def matmul_device(A, B, C, M, K, N):
    """
    Перемножает матрицы A (M x K) и B (K x N), результат записывает в C (M x N).
    Все матрицы - cuda.local.array или cuda.shared.array, тип float32.
    """
    for i in range(M):
        for j in range(N):
            s = 0.0
            for k in range(K):
                s += A[i, k] * B[k, j]
            C[i, j] = s
           

@cuda.jit
def change_acceleration(
        n_nodes_2d: int,
        global_n_elems: int,
        d_global_displacement_x: FloatN,
        d_global_displacement_y: FloatN,
        d_global_acceleration_x: FloatN,
        d_global_acceleration_y: FloatN,
        d_global_elems: IntExS2,
        d_global_mass: FloatN,
        d_global_weights: FloatExS2,
        d_global_nablas: FloatExS2xS2x2,   
        d_hooke: Float3x3,
    ) -> None:
    
    """ 
        Получить целевой eid и целевой la (индекс узла, на который воздействует внутренняя сила)
    """

    la = cuda.threadIdx.x
    eid = cuda.blockIdx.x

    if la >= n_nodes_2d or eid > global_n_elems:
        return

    elem_accelerations_x = cuda.shared.array(128, float32)
    elem_accelerations_y = cuda.shared.array(128, float32)

    elem_accelerations_x[la] = 0
    elem_accelerations_y[la] = 0

    cuda.syncthreads() 
    
    for mu in range(n_nodes_2d):
        for nu in range(n_nodes_2d):

            nabla_la = d_global_nablas[eid, nu, la]
            nabla_mu = d_global_nablas[eid, nu, mu]

            D_la = cuda.local.array((3, 2), float32)

            D_la[0, 0] = nabla_la[0]; D_la[0, 1] = 0.0
            D_la[1, 0] = 0.0;        D_la[1, 1] = nabla_la[1]
            D_la[2, 0] = nabla_la[1]; D_la[2, 1] = nabla_la[0]

            D_la_T = cuda.local.array((2, 3), float32)

            # Транспонирование D_la (3x2 -> 2x3)
            for i in range(3):
                for j in range(2):
                    D_la_T[j, i] = D_la[i, j]

            D_mu = cuda.local.array((3, 2), float32)

            D_mu[0, 0] = nabla_mu[0]; D_mu[0, 1] = 0.0
            D_mu[1, 0] = 0.0;        D_mu[1, 1] = nabla_mu[1]
            D_mu[2, 0] = nabla_mu[1]; D_mu[2, 1] = nabla_mu[0]

            Dtmp1 = cuda.local.array((2, 3), float32)

            K_matrix = cuda.local.array((2, 2), float32)

            matmul_device(D_la_T, d_hooke, Dtmp1, 2, 3, 3)
            matmul_device(Dtmp1, D_mu, K_matrix, 2, 3, 2)

            coeff  = d_global_weights[eid][nu] / d_global_mass[d_global_elems[eid][la]]
            disp_x = d_global_displacement_x[d_global_elems[eid][mu]]
            disp_y = d_global_displacement_y[d_global_elems[eid][mu]]

            elem_accelerations_x[la] -= coeff * (disp_x*K_matrix[0,0] + disp_y*K_matrix[0,1])
            elem_accelerations_y[la] -= coeff * (disp_x*K_matrix[1,0] + disp_y*K_matrix[1,1])

    cuda.atomic.add(d_global_acceleration_x, d_global_elems[eid][la], elem_accelerations_x[la])
    cuda.atomic.add(d_global_acceleration_y, d_global_elems[eid][la], elem_accelerations_y[la])



# @njit
def simulation_loop(
        global_n_nodes: int,
        global_n_elems: int,
        n_nodes_2d: int,
        tau: np.float32,
        step: int,
        riker_pulse_node_id: int, 
        d_riker_pulse: FloatT,
        d_global_displacement_x: FloatN,
        d_global_displacement_y: FloatN,
        d_global_velocity_x: FloatN,
        d_global_velocity_y: FloatN,
        d_global_acceleration_x: FloatN,
        d_global_acceleration_y: FloatN,
        d_global_elems: IntExS2,
        d_global_mass: FloatN,
        d_hooke: Float3x3,
        d_global_weights: FloatExS2,
        d_global_nablas: FloatExS2xS2x2, 
    ) -> None:

    """ получаем eid элемента, с которым собираемся работать """
    

    change_acceleration[global_n_elems, THREADS_COUNT](
        n_nodes_2d, 
        global_n_elems, 
        d_global_displacement_x, 
        d_global_displacement_y, 
        d_global_acceleration_x,
        d_global_acceleration_y,
        d_global_elems,
        d_global_mass,
        d_global_weights,
        d_global_nablas,   
        d_hooke
    )

    blocks_per_grid = (global_n_nodes + THREADS_COUNT - 1) // THREADS_COUNT

    change_velocity_and_displacement[blocks_per_grid, THREADS_COUNT](
        global_n_nodes, 
        tau,
        step,
        riker_pulse_node_id, 
        d_riker_pulse,
        d_global_displacement_x, 
        d_global_displacement_y, 
        d_global_velocity_x, 
        d_global_velocity_y, 
        d_global_acceleration_x,
        d_global_acceleration_y,
    )


def compute_global_mass(constant_density: float, global_coords: FloatNx2, global_elems: IntExS2, nabla_shape_d2:FloatS2xS2x2, weights_d2: FloatS2) -> FloatN:

    global_n_nodes = global_coords.shape[0]
    
    density = np.full((global_n_nodes), constant_density, dtype=np.float64)

    global_mass = np.zeros_like(density)
    
    for elem in global_elems:
        yacobians = compute_yacobians(global_coords[elem], nabla_shape_d2)
        global_mass[elem] += weights_d2*density[elem]*yacobians

    return global_mass


def compute_global_volumes(global_coords: FloatNx2, global_elems: IntExS2, nabla_shape_d2:FloatS2xS2x2, weights_d2: FloatS2) -> FloatN:

    nodes_count = global_coords.shape[0]
    Volume = np.zeros((nodes_count))

    for elem in global_elems:
        Yacobian = compute_yacobians(global_coords[elem], nabla_shape_d2)
        Volume[elem] += weights_d2*Yacobian

    return Volume

def compute_hooke_constants(E: float, nu: float) -> Float3x3:

    # Плоская деформация
    a = E*(1-nu)/(1+nu)/(1-2*nu)
    b = E*nu/(1+nu)/(1-2*nu)
    c = E/2/(1+nu)

    # Плоское напряжение
    # a = E/(1-nu**2)
    # b = E*nu/(1-nu**2)
    # c = E/2/(1+nu)

    return np.array([[a,b,0],[b,a,0],[0,0,c]])



def main() -> None:

    # Порядок спектрального элемента
    n_deg = 5

    # Физический размер пластины
    plate_size = (20, 10)

    # Размер (квадратного) элемента
    single_element_size = 1

    # Сремя симуляции (сек)
    total_simulation_time = 0.5

    # Количество шагов симуляции
    total_steps = 200

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
    riker_pulse_frequency = 10

    # 
    riker_pulse = compute_riker_pulse(riker_pulse_frequency, riker_pulse_amplitude, 0, total_simulation_time, tau)

    x_nodes_count = (grid_size[0]*n_deg + 1)
    y_nodes_count = (grid_size[1]*n_deg + 1)

    # Индекс узла, к которому будет применено возмущение
    pulse_node_id = n_deg*(riker_pulse_point[0] + x_nodes_count*riker_pulse_point[1])
    
    global_n_nodes = global_nodes.shape[0]
    global_n_elems = global_elems.shape[0]
    n_nodes_2d = weights_d2.shape[0]

    global_mass = compute_global_mass(constant_density, global_nodes, global_elems, nabla_shape_d2, weights_d2)

    global_displacement_x = np.zeros((global_n_nodes))
    global_displacement_y = np.zeros((global_n_nodes))

    global_velocity_x = np.zeros((global_n_nodes))
    global_velocity_y = np.zeros((global_n_nodes))

    global_acceleration_x = np.zeros((global_n_nodes))
    global_acceleration_y = np.zeros((global_n_nodes))

    hooke = compute_hooke_constants(constan_young, constan_poisson)

    global_nablas:FloatExS2xS2x2 = np.zeros([global_n_elems, n_nodes_2d, n_nodes_2d, 2])
    global_weights: FloatExS2 = np.zeros([global_n_elems, n_nodes_2d])

    for eid, elem in enumerate(global_elems):
        yacobians = compute_yacobians(global_nodes[elem], nabla_shape_d2)
        antiyacobi_ms = compute_antiyacobi_ms(global_nodes[elem], nabla_shape_d2)

        global_nablas[eid] = nabla_shape_d2@antiyacobi_ms
        global_weights[eid] = weights_d2*yacobians
        pass

    d_global_displacement_x: DeviceNDArray = cuda.to_device(global_displacement_x.astype(np.float32))
    d_global_velocity_x: DeviceNDArray = cuda.to_device(global_velocity_x.astype(np.float32))
    d_global_acceleration_x: DeviceNDArray = cuda.to_device(global_acceleration_x.astype(np.float32))
    d_global_displacement_y: DeviceNDArray = cuda.to_device(global_displacement_y.astype(np.float32))
    d_global_velocity_y: DeviceNDArray = cuda.to_device(global_velocity_y.astype(np.float32))
    d_global_acceleration_y: DeviceNDArray = cuda.to_device(global_acceleration_y.astype(np.float32))

    d_global_elems: DeviceNDArray = cuda.to_device(global_elems.astype(np.int32))
    d_global_mass: DeviceNDArray = cuda.to_device(global_mass.astype(np.int32))
    d_global_weights: DeviceNDArray = cuda.to_device(global_weights.astype(np.float32))
    d_global_nablas: DeviceNDArray = cuda.to_device(global_nablas.astype(np.float32))

    d_hooke = cuda.to_device(hooke.astype(np.float32))
    d_riker_pulse = cuda.to_device(riker_pulse.astype(np.float32))

    if VISUAL_MODE:

        visual_field = np.zeros((y_nodes_count, x_nodes_count))

        fig, ax = plt.subplots()
        im = ax.imshow(visual_field, cmap='viridis', vmin=0, vmax=0.0001)

        def update(step: int) -> list[Any]:
            step_duration = time.time()

            simulation_loop(
                global_n_nodes,
                global_n_elems,
                n_nodes_2d,
                np.float32(tau),
                step,
                pulse_node_id, 
                d_riker_pulse,
                d_global_displacement_x, 
                d_global_displacement_y, 
                d_global_velocity_x, 
                d_global_velocity_y, 
                d_global_acceleration_x, 
                d_global_acceleration_y, 
                d_global_elems,
                d_global_mass,
                d_hooke,
                d_global_weights,
                d_global_nablas
            )

            global_velocity_x = d_global_velocity_x.copy_to_host()
            global_velocity_y = d_global_velocity_y.copy_to_host()

            visual_field = ((global_velocity_x**2 + global_velocity_y**2)).reshape(y_nodes_count, x_nodes_count)

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
                global_n_nodes,
                global_n_elems,
                n_nodes_2d,
                np.float32(tau),
                step,
                pulse_node_id, 
                d_riker_pulse,
                d_global_displacement_x, 
                d_global_displacement_y, 
                d_global_velocity_x, 
                d_global_velocity_y, 
                d_global_acceleration_x, 
                d_global_acceleration_y, 
                d_global_elems,
                d_global_mass,
                d_hooke,
                d_global_weights,
                d_global_nablas
            )

            print(step, time.time() - step_duration)


if __name__ == '__main__':

    main()

