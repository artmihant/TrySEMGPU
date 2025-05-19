import time

from typing import Any
import numpy as np
from numba import njit

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from const import *
from test_mesh import generate_spectral_mesh_2d_box

from numba import cuda, float32

def compute_riker_pulse(f:float, A:float, t0:float, t1:float, dt:float) -> FloatT:
    return \
        (lambda t: 
            (lambda p: A*(1+2*p)*np.exp(p))
            (-(np.pi*(f*(t-t0)-1))**2)
        )(np.arange(t0, t1, dt))


@cuda.jit
def change_acceleration(
        d_global_acceleration: DFloatNxD, 
        # dc_element_types_register: DIntA,
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

    if eid >= elements_count:
        return

    # dc_element_types_nabla_shapes = cuda.const.array_like(element_types_nabla_shapes)
    # dc_element_types_weights = cuda.const.array_like(element_types_weights)

    # TODO: Add reading from element_types_register
    element_family = d_elements_families[eid]
    element_dim = d_elements_dims[eid]
    element_deg = d_elements_degs[eid]

    offset = d_elements_offsets[eid]
    
    nodes_count = d_elements_offsets[eid+1]-offset

    nu = cuda.threadIdx.x

    if nu >= nodes_count:
        return

    nid = d_elements_nids[offset+nu]

    element_weights = d_element_types_weights #const
    element_nabla_shapes  = d_element_types_nabla_shapes #const

    element_young = cuda.shared.array(128, dtype=float32)
    element_young[nu] = d_global_young[offset+nu, 0]

    element_poisson = cuda.shared.array(THREADS_COUNT, FLOAT)
    element_poisson[nu] = d_global_poisson[offset+nu, 0]

    element_displacement = cuda.shared.array((THREADS_COUNT, 2), FLOAT)
    element_displacement[nu, 0] = d_global_displacement[nid, 0]
    element_displacement[nu, 1] = d_global_displacement[nid, 1]

    element_nodes = cuda.shared.array((THREADS_COUNT, 2), FLOAT)
    element_nodes[nu, 0] = d_global_nodes[nid, 0]
    element_nodes[nu, 1] = d_global_nodes[nid, 1]

    elem_accelerations = cuda.shared.array((THREADS_COUNT, 2), FLOAT)
    elem_accelerations[nu, 0] = 0
    elem_accelerations[nu, 1] = 0

    cuda.syncthreads() 

    for la in range(nodes_count):
        
        yacobi_0_0 = 0
        yacobi_0_1 = 0
        yacobi_1_0 = 0
        yacobi_1_1 = 0

        for mu in range(nodes_count):
            yacobi_0_0 += element_nodes[mu,0]*element_nabla_shapes[la,mu,0]
            yacobi_0_1 += element_nodes[mu,0]*element_nabla_shapes[la,mu,1]
            yacobi_1_0 += element_nodes[mu,1]*element_nabla_shapes[la,mu,0]
            yacobi_1_1 += element_nodes[mu,1]*element_nabla_shapes[la,mu,1]

        yacobian = yacobi_0_0*yacobi_1_1 - yacobi_0_1*yacobi_1_0

        inv_yacobian = 1.0 / yacobian

        inv_yacobi_0_0 =  yacobi_1_1 * inv_yacobian
        inv_yacobi_0_1 = -yacobi_0_1 * inv_yacobian
        inv_yacobi_1_0 = -yacobi_1_0 * inv_yacobian
        inv_yacobi_1_1 =  yacobi_0_0 * inv_yacobian

        weights = element_weights[la]*yacobian

        young = element_young[la]
        poisson = element_poisson[la]

        hooke_a = young*(1-poisson)/(1+poisson)/(1-2*poisson)
        hooke_b = young*poisson/(1+poisson)/(1-2*poisson)
        hooke_c = young/2/(1+poisson)

        for mu in range(nodes_count):

            L_n_l_0 = element_nabla_shapes[la,nu,0]*inv_yacobi_0_0 + element_nabla_shapes[la,nu,1]*inv_yacobi_1_0
            L_n_l_1 = element_nabla_shapes[la,nu,0]*inv_yacobi_0_1 + element_nabla_shapes[la,nu,1]*inv_yacobi_1_1

            L_m_l_0 = element_nabla_shapes[la,mu,0]*inv_yacobi_0_0 + element_nabla_shapes[la,mu,1]*inv_yacobi_1_0
            L_m_l_1 = element_nabla_shapes[la,mu,0]*inv_yacobi_0_1 + element_nabla_shapes[la,mu,1]*inv_yacobi_1_1

            k_block_0_0 = hooke_a*L_n_l_0*L_m_l_0 + hooke_c*L_n_l_1*L_m_l_1
            k_block_0_1 = hooke_b*L_n_l_0*L_m_l_1 + hooke_c*L_n_l_1*L_m_l_0
            k_block_1_0 = hooke_b*L_n_l_1*L_m_l_0 + hooke_c*L_n_l_0*L_m_l_1
            k_block_1_1 = hooke_a*L_n_l_1*L_m_l_1 + hooke_c*L_n_l_0*L_m_l_0

            elem_accelerations[nu, 0] -= weights*(k_block_0_0*element_displacement[mu, 0] + k_block_0_1*element_displacement[mu, 1]) / d_global_mass[nid, 0]
            elem_accelerations[nu, 1] -= weights*(k_block_1_0*element_displacement[mu, 0] + k_block_1_1*element_displacement[mu, 1]) / d_global_mass[nid, 0]

    cuda.atomic.add(d_global_acceleration, (nid, 0), elem_accelerations[nu, 0])
    cuda.atomic.add(d_global_acceleration, (nid, 1), elem_accelerations[nu, 1])

    # cuda.atomic.add(d_global_acceleration, (1, 1), 1)


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
        # dc_element_types_register: DIntA,
        d_element_types_nabla_shapes: FloatSxSxD,
        d_element_types_weights: FloatS,
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
        # dc_element_types_register,
        d_element_types_nabla_shapes,
        d_element_types_weights,
        d_elements_offsets, 
        d_elements_families,
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
    full_duration = time.time()
    print('hello')

    # Порядок спектрального элемента
    n_deg = 7

    # Физический размер пластины
    plate_size = (40, 20)

    # Размер (квадратного) элемента
    single_element_size = 1

    # Сремя симуляции (сек)
    total_simulation_time = 1

    # Количество шагов симуляции
    total_steps = 800

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
    print(time.time() - full_duration)
    full_duration = time.time()
    print('make task')

    # Генератор сетки
    mesh, elements_type = generate_spectral_mesh_2d_box(grid_size, single_element_size, start_point, n_deg)
    
    x_nodes_count = (grid_size[0]*n_deg + 1)
    y_nodes_count = (grid_size[1]*n_deg + 1)
    print(time.time() - full_duration)
    full_duration = time.time()
    print('upload data')

    # print('compute_k_matrix')

    # for element in mesh.elements:
    #     element.compute_k_matrix(constan_young, constan_poisson)

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
    d_element_types_nabla_shapes = cuda.to_device(elements_type.nabla_shape)
    
    element_types_weights = elements_type.weights
    d_element_types_weights = cuda.to_device(element_types_weights)
    
    element_types_register = np.array([1, 2, n_deg, 0, 0], INT)
    # dc_element_types_register = cuda.const.array_like(element_types_register)

    # print('compute global_mass')

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


            # if step < riker_pulse.shape[0]:
            #     d_global_outer_forces[riker_pulse_nid:(riker_pulse_nid+1), 1:2].copy_to_device(np.array([[riker_pulse[step]]], dtype=np.float32))
            

            simulation_step(
                # element_types_register,
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
        
        print(time.time() - full_duration)
        full_duration = time.time()

        print('simulation start')

        for step in range(total_steps):
            step_duration = time.time()

            if step < riker_pulse.shape[0]:
                global_outer_forces[riker_pulse_nid][1] = riker_pulse[step]
                d_global_outer_forces.copy_to_device(global_outer_forces)
            
            # if step < riker_pulse.shape[0]:
            #     d_global_outer_forces[riker_pulse_nid:(riker_pulse_nid+1), 1:2].copy_to_device(np.array([[riker_pulse[step]]], dtype=np.float32))
            
            simulation_step(
                # dc_element_types_register,
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
            print(step, time.time() - step_duration)

        print(time.time() - full_duration)
        print('success!')

if __name__ == '__main__':

    main()

