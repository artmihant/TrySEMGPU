from const import *
from spectral_elements import SpectralElement, SpectralElementType, SpectralMesh
from itertools import product


def generate_spectral_mesh_2d_box(grid_size: tuple[int, int], single_element_size: float, start_point: tuple[float, float], n_deg: int) -> SpectralMesh:

    elements_type = SpectralElementType('cube', DIM, n_deg)
    elements_type_1d = SpectralElementType('cube', 1, n_deg)

    x_nodes_count = (grid_size[0]*n_deg + 1)
    y_nodes_count = (grid_size[1]*n_deg + 1)

    nodes_count = x_nodes_count*y_nodes_count

    n_nodes_1d = n_deg + 1
    n_nodes_2d = n_nodes_1d**2

    nodes = np.zeros((nodes_count, DIM), dtype=FLOAT)

    elements = []

    for ey, ex in product(range(grid_size[1]), range(grid_size[0])):

        element_nids = np.zeros(n_nodes_2d, INT)

        for sy, sx in product(range(n_nodes_1d), range(n_nodes_1d)):

            nx = sx + n_deg*ex 
            ny = sy + n_deg*ey

            node_index = nx + x_nodes_count*ny

            nodes[node_index] = [
                start_point[0] + (ex + (elements_type_1d.xi_nodes[sx]+1)*0.5 ) * single_element_size,
                start_point[1] + (ey + (elements_type_1d.xi_nodes[sy]+1)*0.5 ) * single_element_size
            ]

            spectral_index = sx + n_nodes_1d*sy

            element_nids[spectral_index] = node_index

        element = SpectralElement(nodes, element_nids, elements_type)

        elements.append(element)

    return SpectralMesh(nodes, elements)
