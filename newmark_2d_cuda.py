import time

from typing import Any
import numpy as np
from numba import njit

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from const import *
from test_mesh import generate_spectral_mesh_2d_box


def compute_riker_pulse(f:float, A:float, t0:float, t1:float, dt:float) -> FloatT:
    return \
        (lambda t: 
            (lambda p: A*(1+2*p)*np.exp(p))
            (-(np.pi*(f*(t-t0)-1))**2)
        )(np.arange(t0, t1, dt))

@njit
def simulation_step(
        element_types_register: IntA,
        element_types_nabla_shapes: FloatSxSxD,
        element_types_weights: FloatS,
        elements_offsets: IntA, 
        elements_families: IntE, 
        elements_dims: IntE, 
        elements_degs: IntE, 
        elements_nids: IntEN,
        global_nodes: FloatNxD,
        global_mass: FloatNx1,
        global_young: FloatENx1,      
        global_poisson: FloatENx1,
        global_outer_force: FloatNxD, 
        global_displacement: FloatNxD, 
        global_velocity: FloatNxD, 
        global_acceleration: FloatNxD, 
        tau: float,
    ):

    global_acceleration[:] = 0
    global_acceleration += global_outer_force/global_mass

    elements_count = len(elements_families)

    for eid in range(elements_count):
        
        # TODO: Add reading from element_types_register
        element_family = elements_families[eid]
        element_dim = elements_dims[eid]
        element_deg = elements_degs[eid]
        element_weights = element_types_weights
        element_nabla_shapes = element_types_nabla_shapes

        offset_left = elements_offsets[eid]
        offset_right = elements_offsets[eid+1]

        element_young:FloatSx1 = global_young[offset_left:offset_right]
        element_poisson:FloatSx1 = global_poisson[offset_left:offset_right]

        element_nids: IntS = elements_nids[offset_left:offset_right]

        nodes_count = len(element_nids)
        elem_inner_force = np.zeros((len(element_nids), 2), FLOAT)

        element_displacement:FloatSxD = global_displacement[element_nids]

        element_nodes:FloatSxD = global_nodes[element_nids]


        for la in range(nodes_count):

            yacobi = element_nodes.transpose()@element_nabla_shapes[la]
            antiyacobi = np.linalg.inv(yacobi)
            weights = element_weights[la]*np.linalg.det(yacobi)

            for nu in range(nodes_count):

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

                    elem_inner_force[nu] += weights*k_block@element_displacement[mu]

        global_acceleration[element_nids] -= elem_inner_force / global_mass[element_nids]

    global_velocity += tau*global_acceleration
    global_displacement += tau*global_velocity


def main():

    # Порядок спектрального элемента
    n_deg = 3

    # Физический размер пластины
    plate_size = (10, 5)

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

    global_poisson = mesh.nids_array(1)
    global_poisson.fill(constan_poisson)

    global_mass = mesh.nodes_array(1)

    for eid, element in enumerate(mesh.elements):
        global_mass[element.nids] += element.weights*global_density[mesh.offsets[eid]:mesh.offsets[eid+1]]

    global_displacement = mesh.nodes_array(DIM)
    global_velocity = mesh.nodes_array(DIM)
    global_acceleration = mesh.nodes_array(DIM)
    global_outer_force = mesh.nodes_array(DIM)

    global_nodes = mesh.nodes
    elements_nids = mesh.nids
    elements_families = mesh.get_elements_families()
    elements_offsets = mesh.offsets
    elements_dims = mesh.get_elements_dims()
    elements_degs = mesh.get_elements_degs()
    
    element_types_nabla_shapes = elements_type.nabla_shape
    element_types_weights = elements_type.weights

    element_types_register = np.array([1, 2, n_deg, 0, 0], INT)

    if VISUAL_MODE:

        visual_field = np.zeros((y_nodes_count, x_nodes_count), dtype=FLOAT)

        fig, ax = plt.subplots()
        im = ax.imshow(visual_field, cmap='viridis', vmin=0, vmax=0.001)
        
        full_duration = time.time()
        
        def update(step: int) -> list[Any]:
            step_duration = time.time()

            if step < riker_pulse.shape[0]:
                global_outer_force[riker_pulse_nid][1] = riker_pulse[step]

            simulation_step(
                element_types_register,
                element_types_nabla_shapes,
                element_types_weights,
                elements_offsets,
                elements_families, 
                elements_dims,
                elements_degs,
                elements_nids, 
                global_nodes,
                global_mass,
                global_young,         
                global_poisson,  
                global_outer_force, 
                global_displacement, 
                global_velocity, 
                global_acceleration, 
                tau
            )

            visual_field = (global_mass[:, 0]*(global_velocity[:, 0]**2 + global_velocity[:, 1]**2)).reshape(y_nodes_count, x_nodes_count)

            print(step, time.time() - step_duration, time.time() - full_duration, np.sum(visual_field), np.max(visual_field))
            
            im.set_data(visual_field)  # обновляем данные среза
            # im.set_clim(vmin=visual_field.min(), vmax=visual_field.max())

            return [im]

        ani = FuncAnimation(fig, update, interval=10)
        plt.show()

    else:
        
        for step in range(total_steps):
            step_duration = time.time()

            if step < riker_pulse.shape[0]:
                global_outer_force[riker_pulse_nid][1] = riker_pulse[step]

            simulation_step(
                element_types_register,
                element_types_nabla_shapes,
                element_types_weights,
                elements_offsets,
                elements_families, 
                elements_dims,
                elements_degs,
                elements_nids, 
                global_nodes,
                global_mass,
                global_young,         
                global_poisson,  
                global_outer_force, 
                global_displacement, 
                global_velocity, 
                global_acceleration, 
                tau
            )

            print(step, time.time() - step_duration)


if __name__ == '__main__':

    main()

