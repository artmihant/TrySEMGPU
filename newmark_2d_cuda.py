import time

from typing import Any
import numpy as np
from numba import njit

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from const import *
from test_mesh import generate_spectral_mesh_2d_box

from numba import cuda

def compute_riker_pulse(f:float, A:float, t0:float, t1:float, dt:float) -> FloatT:
    return \
        (lambda t: 
            (lambda p: A*(1+2*p)*np.exp(p))
            (-(np.pi*(f*(t-t0)-1))**2)
        )(np.arange(t0, t1, dt))


@cuda.jit
def change_acceleration(
        d_global_acceleration: DFloatNxD, 
        d_element_types_register: DIntA,
        d_element_types_nabla_shapes: DFloatSxSxD,
        d_element_types_weights: DFloatS,
        d_elements_offsets: DIntA, 
        d_elements_families: DIntE, 
        d_elements_dims: DIntE, 
        d_elements_degs: DIntE, 
        d_elements_nids: DIntEN, 
        d_global_nodes: DFloatNxD,       
        d_global_displacement: DFloatNxD,  
        d_global_mass: DFloatNx1,
        d_global_young: DFloatENx1,      
        d_global_poisson: DFloatENx1,
    ):
    eid = cuda.blockIdx.x

    elements_count = d_elements_dims.shape[0]

    if eid > elements_count:
        return

    # TODO: Add reading from element_types_register
    element_family = d_elements_families[eid]
    element_dim = d_elements_dims[eid]
    element_deg = d_elements_degs[eid]

    offset_left = d_elements_offsets[eid]
    offset_right = d_elements_offsets[eid+1]
    
    nodes_count = offset_right-offset_left

    nu = cuda.threadIdx.x

    if nu >= nodes_count:
        return

    nid = d_elements_nids[offset_left+nu]

    element_weights = d_element_types_weights #const
    element_nabla_shapes  = d_element_types_nabla_shapes #const

    element_young = d_global_young[offset_left:offset_right] #share
    element_poisson = d_global_poisson[offset_left:offset_right] #share

    element_displacement = d_global_displacement[element_nids] #share

    element_nodes:FloatSxD = d_global_nodes[element_nids] #share

    elem_accelerations = np.zeros((nodes_count, 2), FLOAT) #share

    for la in range(nodes_count):

        yacobi = element_nodes.transpose()@element_nabla_shapes[la]
        antiyacobi = np.linalg.inv(yacobi)
        weights = element_weights[la]*np.linalg.det(yacobi)

        for mu in range(nodes_count):

            young = element_young[la][0]
            poisson = element_poisson[la][0]

            hooke_a = young*(1-poisson)/(1+poisson)/(1-2*poisson)
            hooke_b = young*poisson/(1+poisson)/(1-2*poisson)
            hooke_c = young/2/(1+poisson)

            L_n_l = element_nabla_shapes[la][nu]@antiyacobi
            L_m_l = element_nabla_shapes[la][mu]@antiyacobi

            k_block = np.array([
                [   
                    hooke_a*L_n_l[0]*L_m_l[0] + hooke_c*L_n_l[1]*L_m_l[1], 
                    hooke_b*L_n_l[0]*L_m_l[1] + hooke_c*L_n_l[1]*L_m_l[0]
                ],
                [
                    hooke_b*L_n_l[1]*L_m_l[0] + hooke_c*L_n_l[0]*L_m_l[1], 
                    hooke_a*L_n_l[1]*L_m_l[1] + hooke_c*L_n_l[0]*L_m_l[0]
                ]
            ], dtype=FLOAT)

            elem_accelerations[nu] -= weights*k_block@element_displacement[mu] / d_global_mass[nid]

    cuda.atomic.add(d_global_acceleration, (nid, 0), elem_accelerations[nu, 0])
    cuda.atomic.add(d_global_acceleration, (nid, 1), elem_accelerations[nu, 1])




@cuda.jit
def change_velocity_and_displacement(
        tau: float, 
        d_global_outer_force: FloatNxD, 
        d_global_displacement: FloatNxD, 
        d_global_velocity: FloatNxD, 
        d_global_acceleration: FloatNxD,
    ) -> None:

    nid = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    nodes_count = d_global_displacement.shape[0]

    if nid >= nodes_count:
        return

    d_global_velocity[nid, 0] += tau*d_global_acceleration[nid, 0]
    d_global_velocity[nid, 1] += tau*d_global_acceleration[nid, 1]
    d_global_displacement[nid, 0] += tau*d_global_velocity[nid, 0]
    d_global_displacement[nid, 1] += tau*d_global_velocity[nid, 1]

    d_global_acceleration[nid, 0] = d_global_outer_force[nid, 0]
    d_global_acceleration[nid, 1] = d_global_outer_force[nid, 1]


def simulation_step(
        d_element_types_register: DIntA,
        d_element_types_nabla_shapes: DFloatSxSxD,
        d_element_types_weights: DFloatS,
        d_elements_offsets: DIntA, 
        d_elements_families: DIntE, 
        d_elements_dims: DIntE, 
        d_elements_degs: DIntE, 
        d_elements_nids: DIntEN,
        d_global_nodes: DFloatNxD,
        d_global_mass: DFloatNx1,
        d_global_young: DFloatENx1,      
        d_global_poisson: DFloatENx1,
        d_global_outer_force: DFloatNxD, 
        d_global_displacement: DFloatNxD, 
        d_global_velocity: DFloatNxD, 
        d_global_acceleration: DFloatNxD, 
        tau: float,
    ):

    elements_count = d_elements_families.shape[0]
    nodes_count = d_global_nodes.shape[0]


    change_acceleration[elements_count, THREADS_COUNT](
        d_global_acceleration,
        d_element_types_register,
        d_element_types_nabla_shapes,
        d_element_types_weights,
        d_elements_families,
        d_elements_offsets, 
        d_elements_dims, 
        d_elements_degs, 
        d_elements_nids,    
        d_global_nodes,    
        d_global_displacement,
        d_global_mass,
        d_global_young,      
        d_global_poisson,
    )

    blocks_per_grid = (nodes_count + THREADS_COUNT - 1) // THREADS_COUNT

    change_velocity_and_displacement[blocks_per_grid, THREADS_COUNT](
        tau, 
        d_global_outer_force, 
        d_global_displacement, 
        d_global_velocity, 
        d_global_acceleration,
    )



def main():

    # Порядок спектрального элемента
    n_deg = 5

    # Физический размер пластины
    plate_size = (20, 10)

    # Размер (квадратного) элемента
    single_element_size = 1

    # Сремя симуляции (сек)
    total_simulation_time = 0.5

    # Количество шагов симуляции
    total_steps = 400

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

    # Генератор сетки
    mesh, elements_type = generate_spectral_mesh_2d_box(grid_size, single_element_size, start_point, n_deg)
    
    x_nodes_count = (grid_size[0]*n_deg + 1)
    y_nodes_count = (grid_size[1]*n_deg + 1)

    for element in mesh.elements:
        element.compute_k_matrix(constan_young, constan_poisson)

    # Точка возмущения Рикера
    riker_pulse_amplitude = 100
    riker_pulse_frequency = 5

    riker_pulse = compute_riker_pulse(riker_pulse_frequency, riker_pulse_amplitude, 0, total_simulation_time, tau)

    riker_pulse_point = [grid_size[0]//2, grid_size[1]]
    riker_pulse_nid = n_deg*(riker_pulse_point[0] + x_nodes_count*riker_pulse_point[1])

    global_density = mesh.nids_array(1)
    global_density.fill(constant_density)

    global_young = mesh.nids_array(1)
    global_young.fill(constan_young)
    d_global_young = cuda.to_device(global_young)

    global_poisson = mesh.nids_array(1)
    global_poisson.fill(constan_poisson)
    d_global_poisson = cuda.to_device(global_poisson)

    global_mass = mesh.nodes_array(1)

    for eid, element in enumerate(mesh.elements):
        global_mass[element.nids] += element.weights*global_density[mesh.offsets[eid]:mesh.offsets[eid+1]]

    d_global_mass = cuda.to_device(global_mass)

    global_displacement = mesh.nodes_array(DIM)
    d_global_displacement = cuda.to_device(global_displacement)
    
    global_velocity = mesh.nodes_array(DIM)
    d_global_velocity = cuda.to_device(global_velocity)

    global_acceleration = mesh.nodes_array(DIM)
    d_global_acceleration = cuda.to_device(global_acceleration)

    global_outer_forces = mesh.nodes_array(DIM)
    d_global_outer_forces = cuda.to_device(global_outer_forces)

    global_nodes = mesh.nodes
    d_global_nodes = cuda.to_device(global_nodes)

    elements_nids = mesh.nids
    d_elements_nids = cuda.to_device(elements_nids)

    elements_families = mesh.get_elements_families()
    d_elements_families = cuda.to_device(elements_families)

    elements_offsets = mesh.offsets
    d_elements_offsets = cuda.to_device(elements_offsets)

    elements_dims = mesh.get_elements_dims()
    d_elements_dims = cuda.to_device(elements_dims)

    elements_degs = mesh.get_elements_degs()
    d_elements_degs = cuda.to_device(elements_degs)
    
    element_types_nabla_shapes = elements_type.nabla_shape
    d_element_types_nabla_shapes = cuda.to_device(element_types_nabla_shapes)
    
    element_types_weights = elements_type.weights
    d_element_types_weights = cuda.to_device(element_types_weights)
    
    element_types_register = np.array([1, 2, n_deg, 0, 0], INT)
    d_element_types_register = cuda.to_device(element_types_register)

    if VISUAL_MODE:

        visual_field = np.zeros((y_nodes_count, x_nodes_count), dtype=FLOAT)

        fig, ax = plt.subplots()
        im = ax.imshow(visual_field, cmap='viridis', vmin=0, vmax=0.001)
        
        full_duration = time.time()
        
        def update(step: int) -> list[Any]:
            step_duration = time.time()

            if step < riker_pulse.shape[0]:
                global_outer_forces[riker_pulse_nid][1] = riker_pulse[step]

            d_global_outer_forces.copy_to_device(global_outer_forces)

            simulation_step(
                d_element_types_register,
                d_element_types_nabla_shapes,
                d_element_types_weights,
                d_elements_offsets,
                d_elements_families, 
                d_elements_dims,
                d_elements_degs,
                d_elements_nids, 
                d_global_nodes,
                d_global_mass,
                d_global_young,         
                d_global_poisson,  
                d_global_outer_forces, 
                d_global_displacement, 
                d_global_velocity, 
                d_global_acceleration, 
                tau
            )

            global_velocity = d_global_velocity.copy_to_host()

            visual_field = (global_mass[:, 0]*(global_velocity[:, 0]**2 + global_velocity[:, 1]**2)).reshape(y_nodes_count, x_nodes_count)

            print(step, time.time() - step_duration, time.time() - full_duration, np.sum(visual_field), np.max(visual_field))
            
            im.set_data(visual_field)  # обновляем данные среза
            # im.set_clim(vmin=visual_field.min(), vmax=visual_field.max())

            return [im]

        ani = FuncAnimation(fig, update, interval=10)
        plt.show()

    else:
        
        full_duration = time.time()

        for step in range(total_steps):
            # step_duration = time.time()

            if step < riker_pulse.shape[0]:
                global_outer_forces[riker_pulse_nid][1] = riker_pulse[step]

            d_global_outer_forces.copy_to_device(global_outer_forces)
            
            simulation_step(
                d_element_types_register,
                d_element_types_nabla_shapes,
                d_element_types_weights,
                d_elements_offsets,
                d_elements_families, 
                d_elements_dims,
                d_elements_degs,
                d_elements_nids, 
                d_global_nodes,
                d_global_mass,
                d_global_young,         
                d_global_poisson,  
                d_global_outer_forces, 
                d_global_displacement, 
                d_global_velocity, 
                d_global_acceleration, 
                tau
            )

            # print(step, time.time() - step_duration)

        print(time.time() - full_duration)

if __name__ == '__main__':

    main()

