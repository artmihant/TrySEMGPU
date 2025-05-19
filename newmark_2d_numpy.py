import time

from typing import Any
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from const import *
from spectral_elements import SpectralMesh
from test_mesh import generate_spectral_mesh_2d_box


def compute_riker_pulse(f:float, A:float, t0:float, t1:float, dt:float) -> FloatT:
    return \
        (lambda t: 
            (lambda p: A*(1+2*p)*np.exp(p))
            (-(np.pi*(f*(t-t0)-1))**2)
        )(np.arange(t0, t1, dt))


def simulation_step(
        mesh: SpectralMesh, 
        tau: float,
        global_outer_force: FloatNxD, 
        global_displacement: FloatNxD, 
        global_velocity: FloatNxD, 
        global_acceleration: FloatNxD, 
        global_mass: FloatNxD,
    ):

    global_acceleration[:] = 0
    global_acceleration += global_outer_force/global_mass

    for element in mesh.elements:

        elem_inner_force = np.tensordot(element.k_matrix, global_displacement[element.nids], axes=([0,2],[0,1]))

        global_acceleration[element.nids] -= elem_inner_force / global_mass[element.nids]

    global_velocity += tau*global_acceleration
    global_displacement += tau*global_velocity


def main():

    # Порядок спектрального элемента
    n_deg = 7

    # Физический размер пластины
    plate_size = (20, 10)

    # Размер (квадратного) элемента
    single_element_size = 1

    # Сремя симуляции (сек)
    total_simulation_time = 1

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

    # Генератор сетки
    mesh, elements_type = generate_spectral_mesh_2d_box(grid_size, single_element_size, start_point, n_deg)
    
    x_nodes_count = (grid_size[0]*n_deg + 1)
    y_nodes_count = (grid_size[1]*n_deg + 1)

    print('compute_k_matrix')
    full_duration = time.time()
    for element in mesh.elements:
        element.compute_k_matrix(constan_young, constan_poisson)
    print(time.time() - full_duration)

    # Точка возмущения Рикера
    riker_pulse_amplitude = 100
    riker_pulse_frequency = 5

    riker_pulse = compute_riker_pulse(riker_pulse_frequency, riker_pulse_amplitude, 0, total_simulation_time, tau)

    riker_pulse_point = [grid_size[0]//2, grid_size[1]]
    riker_pulse_nid = n_deg*(riker_pulse_point[0] + x_nodes_count*riker_pulse_point[1])

    global_density = mesh.nodes_array(1)
    global_density.fill(constant_density)

    global_mass = mesh.nodes_array(1)

    for element in mesh.elements:
        global_mass[element.nids] += element.weights*global_density[element.nids]

    global_displacement = mesh.nodes_array(DIM)
    global_velocity = mesh.nodes_array(DIM)
    global_acceleration = mesh.nodes_array(DIM)
    global_outer_force = mesh.nodes_array(DIM)


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
                mesh, 
                tau,
                global_outer_force, 
                global_displacement, 
                global_velocity, 
                global_acceleration, 
                global_mass,
            )

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
                global_outer_force[riker_pulse_nid][1] = riker_pulse[step]

            simulation_step(
                mesh, 
                tau,
                global_outer_force, 
                global_displacement, 
                global_velocity, 
                global_acceleration, 
                global_mass,
            )

            # print(step, time.time() - step_duration)

        print(time.time() - full_duration)
if __name__ == '__main__':

    main()

