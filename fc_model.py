import binascii
import json
from base64 import b64decode, b64encode
import os
from typing import Any, Callable, Generic, Optional, TypeVar, TypedDict, List, Dict, Union
import numpy as np
from numpy.typing import NDArray

from numpy import ndarray, dtype, int8, int32, int64, float64

""" Version 3.2 """

class FCElementType(TypedDict):
    name: str
    fc_id: int
    dim: int
    order: int
    nodes: int
    structure: Dict[int, np.ndarray]
    edges: List[List[int]]
    facets: List[List[int]]
    tetras: List[List[int]]


FC_ELEMENT_TYPES: Dict[int,FCElementType] = {
    0: {
        'name': 'NONE',
        'fc_id': 0,
        'dim': 0,
        'order': 0,
        'nodes': 0,
        'edges': [],
        'facets': [],
        'tetras': [],
        'structure': {}
    },
    38: {
        'name': 'LUMPMASS3D',
        'fc_id': 38,
        'dim': 0,
        'order': 1,
        'nodes': 1,
        'edges': [],
        'facets': [],
        'tetras': [],
        'structure': {}
    },
    99: {
        'name': 'POINT2D',
        'fc_id': 99,
        'dim': 0,
        'order': 1,
        'nodes': 1,
        'edges': [],
        'facets': [],
        'tetras': [],
        'structure': {}
    },
    101: {
        'name': 'VERTEX1',
        'fc_id': 101,
        'dim': 0,
        'order': 1,
        'nodes': 1,
        'edges': [],
        'facets': [],
        'tetras': [],
        'structure': {}
    },
    36: {
        'name': 'BEAM2a',
        'fc_id': 36,
        'dim': 1,
        'order': 1,
        'nodes': 2,
        'edges': [[0, 1]],
        'facets': [],
        'tetras': [],
        'structure': {}
    },
    37: {
        'name': 'BEAM3a',
        'fc_id': 37,
        'dim': 1,
        'order': 2,
        'nodes': 3,
        'edges': [[0, 2, 1]],
        'facets': [],
        'tetras': [],
        'structure': {}
    },
    39: {
        'name': 'SPRING3D',
        'fc_id': 39,
        'dim': 1,
        'order': 1,
        'nodes': 2,
        'edges': [[0, 1]],
        'facets': [],
        'tetras': [],
        'structure': {}
    },
    89: {
        'name': 'BEAM2b',
        'fc_id': 89,
        'dim': 1,
        'order': 1,
        'nodes': 2,
        'edges': [[0, 1]],
        'facets': [],
        'tetras': [],
        'structure': {}
    },
    90: {
        'name': 'BEAM3b',
        'fc_id': 90,
        'dim': 1,
        'order': 2,
        'nodes': 3,
        'edges': [[0, 2, 1]],
        'facets': [],
        'tetras': [],
        'structure': {}
    },
    107: {
        'name': 'BAR2',
        'fc_id': 107,
        'dim': 1,
        'order': 1,
        'nodes': 2,
        'edges': [[0, 1]],
        'facets': [],
        'tetras': [],
        'structure': {}
    },
    108: {
        'name': 'BAR3',
        'fc_id': 108,
        'dim': 1,
        'order': 2,
        'nodes': 3,
        'edges': [[0, 2, 1]],
        'facets': [],
        'tetras': [],
        'structure': {}
    },
    109: {
        'name': 'CABEL2',
        'fc_id': 109,
        'dim': 1,
        'order': 1,
        'nodes': 2,
        'edges': [[0, 1]],
        'facets': [],
        'tetras': [],
        'structure': {}
    },
    110: {
        'name': 'CABEL3',
        'fc_id': 110,
        'dim': 1,
        'order': 2,
        'nodes': 3,
        'edges': [[0, 2, 1]],
        'facets': [],
        'tetras': [],
        'structure': {}
    },
   10: {
        'name': 'TRI3',
        'fc_id': 10,
        'dim': 2,
        'order': 1,
        'nodes': 3,
        'edges': [[0, 1, 2, 0]],
        'facets': [[0, 1, 2]],
        'tetras': [],
        'structure': {}
    },
    11: {
        'name': 'TRI6',
        'fc_id': 11,
        'dim': 2,
        'order': 2,
        'nodes': 6,
        'edges': [[0, 3, 1, 4, 2, 5, 0]],
        'facets': [[0, 3, 1, 4, 2, 5]],
        'tetras': [],
        'structure': {}
    },
    12: {
        'name': 'QUAD4',
        'fc_id': 12,
        'dim': 2,
        'order': 1,
        'nodes': 4,
        'edges': [[0, 1, 2, 3, 0]],
        'facets': [[0, 1, 2, 3]],
        'tetras': [],
        'structure': {}
    },
    13: {
        'name': 'QUAD8',
        'fc_id': 13,
        'dim': 2,
        'order': 2,
        'nodes': 8,
        'edges': [[0, 4, 1, 5, 2, 6, 3, 7, 0]],
        'facets': [[0, 4, 1, 5, 2, 6, 3, 7]],
        'tetras': [],
        'structure': {}
    },
    29: {
        'name': 'MITC3',
        'fc_id': 29,
        'dim': 2,
        'order': 1,
        'nodes': 3,
        'edges': [[0, 1, 2, 0]],
        'facets': [[0, 1, 2]],
        'tetras': [],
        'structure': {}
    },
    30: {
        'name': 'MITC6',
        'fc_id': 30,
        'dim': 2,
        'order': 2,
        'nodes': 6,
        'edges': [[0, 3, 1, 4, 2, 5, 0]],
        'facets': [[0, 3, 1, 4, 2, 5]],
        'tetras': [],
        'structure': {}
    },
    31: {
        'name': 'MITC4',
        'fc_id': 31,
        'dim': 2,
        'order': 1,
        'nodes': 4,
        'edges': [[0, 1, 2, 3, 0]],
        'facets': [[0, 1, 2, 3]],
        'tetras': [],
        'structure': {}
    },
    32: {
        'name': 'MITC8',
        'fc_id': 32,
        'dim': 2,
        'order': 2,
        'nodes': 8,
        'edges': [[0, 4, 1, 5, 2, 6, 3, 7, 0]],
        'facets': [[0, 4, 1, 5, 2, 6, 3, 7]],
        'tetras': [],
        'structure': {}
    },
    1: {
        'name': 'TETRA4',
        'fc_id': 1,
        'dim': 3,
        'order': 1,
        'nodes': 4,
        'edges': [[0, 1, 2, 0], [0, 3], [1, 3], [2, 3]],
        'facets': [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]],
        'tetras': [[0, 1, 2, 3]],
        'structure': {}
    },
    2: {
        'name': 'TETRA10',
        'fc_id': 2,
        'dim': 3,
        'order': 2,
        'nodes': 10,
        'edges': [[0, 4, 1, 5, 2, 6, 0], [0, 7, 3], [1, 8, 3], [2, 9, 3]],
        'facets': [[0, 6, 2, 5, 1, 4], [0, 4, 1, 8, 3, 5], [1, 5, 2, 9, 3, 8], [2, 6, 0, 5, 3, 9]],
        'tetras': [],
        'structure': {}
    },
    3: {
        'name': 'HEX8',
        'fc_id': 3,
        'dim': 3,
        'order': 1,
        'nodes': 8,
        'edges': [[0, 1, 2, 3, 0], [4, 5, 6, 7, 4], [0, 4], [1, 5], [2, 6], [3, 7]],
        'facets': [[3, 2, 1, 0], [4, 5, 6, 7], [1, 2, 6, 5], [0, 1, 5, 4], [0, 4, 7, 3], [2, 3, 7, 6]],
        'tetras': [[1, 3, 4, 6], [3, 1, 4, 0], [1, 3, 6, 2], [4, 1, 6, 5], [3, 4, 6, 7]],
        'structure': {}
    },
    4: {
        'name': 'HEX20',
        'fc_id': 4,
        'dim': 3,
        'order': 2,
        'nodes': 20,
        'edges': [[0, 8, 1, 9, 2, 10, 3, 11, 0], [4, 12, 5, 13, 6, 14, 7, 15, 4],
                  [0, 0, 4], [1, 0, 5], [2, 0, 6], [3, 0, 7]],
        'facets': [[3, 10, 2, 9, 1, 8, 0, 11], [4, 12, 5, 13, 6, 14, 7, 15], [1, 9, 2, 18, 6, 13, 5, 17],
                   [0, 8, 1, 17, 5, 12, 4, 16], [0, 16, 4, 15, 7, 19, 3, 11], [2, 10, 3, 19, 7, 14, 6, 18]],
        'tetras': [],
        'structure': {}
    },
    6: {
        'name': 'WEDGE6',
        'fc_id': 6,
        'dim': 3,
        'order': 1,
        'nodes': 5,
        'edges': [[0, 1, 2, 0], [3, 4, 5, 3], [0, 3], [1, 4], [2, 5]],
        'facets': [[0, 1, 2], [5, 4, 3], [0, 2, 5, 3], [0, 3, 4, 1], [1, 4, 5, 2]],
        'tetras': [[0, 5, 4, 3], [0, 4, 2, 1], [0, 2, 4, 5]],
        'structure': {}
    },
    7: {
        'name': 'WEDGE15',
        'fc_id': 7,
        'dim': 3,
        'order': 2,
        'nodes': 15,
        'edges': [[0, 5, 1, 6, 2, 7, 3, 8, 0], [0, 9, 4], [1, 10, 4], [2, 11, 4], [3, 12, 4]],
        'facets': [[3, 7, 2, 6, 1, 5, 0, 8],
                   [0, 5, 1, 10, 4, 9], [1, 6, 2, 11, 4, 10], [2, 7, 3, 12, 4, 11], [3, 8, 0, 9, 4, 12]],
        'tetras': [],
        'structure': {}
    },
    8: {
        'name': 'PYR5',
        'fc_id': 8,
        'dim': 3,
        'order': 1,
        'nodes': 5,
        'edges': [[0, 1, 2, 3, 0], [0, 4], [1, 4], [2, 4], [3, 4]],
        'facets': [[3, 2, 1, 0], [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]],
        'tetras': [[1, 3, 4, 0], [3, 4, 1, 2]],
        'structure': {}
    },
    9: {
        'name': 'PYR13',
        'fc_id': 9,
        'dim': 3,
        'order': 2,
        'nodes': 13,
        'edges': [[0, 5, 1, 6, 2, 7, 3, 8, 0], [0, 9, 4], [1, 10, 4], [2, 11, 4], [3, 12, 4]],
        'facets': [[3, 7, 2, 6, 1, 5, 0, 8],
                   [0, 5, 1, 10, 4, 9], [1, 6, 2, 11, 4, 10], [2, 7, 3, 12, 4, 11], [3, 8, 0, 9, 4, 12]],
        'tetras': [],
        'structure': {}
    }
}


def split_facet(facet: List[int]) -> List[int]:
    if len(facet) == 3:
        return facet
    if len(facet) < 3:
        return []
    tail = facet[2:]
    tail.append(facet[1])
    tris = [facet[-1], facet[0], facet[1]]
    tris.extend(split_facet(tail))
    return tris


def split_edge(edge: List[int]) -> List[int]:
    if len(edge) == 2:
        return edge
    if len(edge) < 2:
        return []
    tail = edge[1:]
    pairs = [edge[0], edge[1]]
    pairs.extend(split_edge(tail))
    return pairs


def split_polihedron(tetra: List[int]) -> List[int]:
    return tetra

def make_structure():
    for eid in FC_ELEMENT_TYPES:
        element_type = FC_ELEMENT_TYPES[eid]
        element_type['structure'][0] = np.arange(element_type['nodes'], dtype=np.int32)

        if element_type['dim'] > 0:

            pairs = []
            for edge in element_type['edges']:
                pairs.extend(split_edge(edge))

            element_type['structure'][1] = np.array(pairs, dtype=np.int32)

        if element_type['dim'] > 1:

            trangles = []
            for facet in element_type['facets']:
                trangles.extend(split_facet(facet))

            element_type['structure'][2] = np.array(trangles, dtype=np.int32)

        if element_type['dim'] > 2:

            tetras = []
            for tetra in element_type['tetras']:
                tetras.extend(split_polihedron(tetra))

            element_type['structure'][3] = np.array(tetras, dtype=np.int32)

make_structure()


class RequiredId(TypedDict):
    id: int


T = TypeVar("T", bound=RequiredId)


class FCDict(Generic[T]):
    data: Dict[int, T]

    max_id:int

    def __init__(self):
        self.max_id = 0
        self.data = {}

    def __getitem__(self, key:int):
        return self.data[key]

    def __setitem__(self, key:int, item: T):
        item['id'] = key
        if self.max_id < item['id']:
            self.max_id = item['id']
        self.data[item['id']] = item

    def __contains__(self, key):
        return key in self.data

    def __iter__(self):
        for item in self.data:
            yield self.data[item]

    def __repr__(self) -> str:
        return f'<FCDict: {len(self.data)}>'

    def add(self, item: T):
        if item['id'] in self or item['id'] < 1:
            self[self.max_id+1] = item
        else:
            self[item['id']] = item
        return item['id']

    def __len__(self):
        return len(self.data)

    def keys(self):
        return self.data.keys()

    def reindex(self, index_map: Dict[int, int]):
        new_data = {}
        for key in index_map:
            if item := self.data[key]:
                item['id'] = index_map[key]
                new_data[index_map[key]] = item

        if len(new_data) > 1:
            self.max_id = max(*new_data.keys())
        elif len(new_data) == 1:
            self.max_id = new_data[0]['id']
        else:
            self.max_id = 0

        self.data = new_data

    def compress(self):
        index_map = {index: i + 1 for i, index in enumerate(self.keys())}
        self.reindex(index_map)
        return index_map



def isBase64(sb):
    if sb == 'all':
        return False
    try:
        if isinstance(sb, str):
            sb_bytes = bytes(sb, 'ascii')
        elif isinstance(sb, bytes):
            sb_bytes = sb
        else:
            raise ValueError("Argument must be string or bytes")
        return b64encode(b64decode(sb_bytes)) == sb_bytes
    except (TypeError, binascii.Error):
        return False


def decode(src: str, dtype:dtype = dtype('int32')) -> NDArray:
    if src == '':
        return np.array([], dtype=dtype) # type: ignore
    return np.frombuffer(b64decode(src), dtype) # type: ignore


def fdecode(src: str, dtype:dtype = dtype('int32')) -> Union[NDArray, str]:
    if src == '':
        return np.array([], dtype=dtype) # type: ignore
    if isBase64(src):
        return decode(src, dtype)
    return src


def encode(data: ndarray) -> str:
    return b64encode(data.tobytes()).decode()


def fencode(data: Union[ndarray,str, int]) -> str:
    if isinstance(data, str):
        return data
    if isinstance(data, int):
        return str(data)
    if isinstance(data, ndarray):
        return encode(data)


class FCHeader(TypedDict):
    binary: bool
    description: str
    version: int
    types: Dict[str, int]


# class FCBlockMaterialSteps(TypedDict):
#     ids: NDArray[int32]
#     steps: NDArray[int32]


class FCBlock(TypedDict):
    id: int
    cs_id: int
    material_id: int
    property_id: int
    # steps: NotRequired[NDArray[int32]]
    # material: NotRequired[FCBlockMaterialSteps]


class FCCoordinateSystem(TypedDict):
    id: int
    type: str
    name: str
    origin: NDArray[float64]
    dir1: NDArray[float64]
    dir2: NDArray[float64]


class FCElem(TypedDict):
    id: int
    block: int
    parent_id: int
    type: FCElementType
    nodes: List[int]
    order: int


class FCNode(TypedDict):
    id: int
    xyz: NDArray[float64]


class FCElems:

    data: Dict[str, FCDict[FCElem]]

    def __init__(self, data=None):
        self.data = {
            fc_type['name']: FCDict() for fc_type in FC_ELEMENT_TYPES.values()
        }

        if data:
            self.decode(data)


    def decode(self, data=None):
        if data is None:
            return

        elem_blocks = decode(data.get('elem_blocks', ''))
        elem_orders = decode(data.get('elem_orders', ''))
        elem_parent_ids = decode(data.get('elem_parent_ids', ''))
        elem_types = decode(data.get('elem_types', ''), dtype('int8'))
        elem_ids = decode(data.get('elemids',''))
        elem_nodes = decode(data.get('elems', ''))

        elem_sizes = np.vectorize(lambda t: FC_ELEMENT_TYPES[t]['nodes'])(elem_types)
        elem_offsets = [0, *np.cumsum(elem_sizes)]

        for i, eid in enumerate(elem_ids):
            fc_type = FC_ELEMENT_TYPES[elem_types[i]]
            assert fc_type['order'] == elem_orders[i]

            self.data[fc_type['name']][eid] = {
                'id': eid,
                'type': fc_type,
                'nodes': elem_nodes[elem_offsets[i]:elem_offsets[i+1]].tolist(),
                'block': elem_blocks[i],
                'order': elem_orders[i],
                'parent_id': elem_parent_ids[i],
            }


    def encode(self):

        elems_count = len(self)

        elem_ids = np.zeros(elems_count, np.int32)
        elem_blocks = np.zeros(elems_count, np.int32)
        elem_orders = np.zeros(elems_count, np.int32)
        elem_parent_ids = np.zeros(elems_count, np.int32)
        elem_types = np.zeros(elems_count, np.int8)

        for i, elem in enumerate(self):
            elem_ids[i] = elem['id']
            elem_blocks[i] = elem['block']
            elem_parent_ids[i] = elem['parent_id']
            elem_orders[i] = elem['order']
            elem_types[i] = elem['type']['fc_id']

        elem_nodes = np.array(self.nodes_list, np.int32)

        return {
            "elem_blocks": encode(elem_blocks),
            "elem_orders": encode(elem_orders),
            "elem_parent_ids": encode(elem_parent_ids),
            "elem_types": encode(elem_types),
            "elemids": encode(elem_ids),
            "elems": encode(elem_nodes),
            "elems_count": elems_count,
        }


    def __len__(self):
        return sum([len(self.data[typename]) for typename in self.data])

    def __bool__(self):
        return len(self) > 0

    def __iter__(self):
        for typename in self.data:
            for elem in self.data[typename]:
                yield elem

    def __contains__(self, key):
        for tp in self.data:
            if key in self.data[tp]:
                return True
        return False

    def __getitem__(self, key:Union[int, str]):
        if isinstance(key, str):
            return self.data[key]
        elif isinstance(key, int):
            for typename in self.data:
                if key in self.data[typename]:
                    return self.data[typename][key]
        raise KeyError(f'{key}')
    
    def __setitem__(self, key:int, item: FCElem):
        self.data[item['type']['name']].add(item)

    @property
    def nodes_list(self):
        return [node for elem in self for node in elem['nodes']]

    def compress(self):
        index_map = {elem['id']: i + 1 for i, elem in enumerate(self)}
        self.reindex(index_map)
        return index_map

    def reindex(self, index_map):
        for typename in self.data:
            self.data[typename].reindex(index_map)

    @property
    def max_id(self):
        max_id = 0
        for tp in self.data:
            if max_id < self.data[tp].max_id:
                max_id = self.data[tp].max_id 
        return max_id

    def add(self, item: FCElem):
        if item['id'] in self or item['id'] < 1:
            item['id'] = self.max_id+1

        return self.data[item['type']['name']].add(item)




class FCDependency(TypedDict):
    type: int
    data: Union[NDArray[float64], str, int]


class FCMaterialProperty(TypedDict):
    type: int
    name: int
    data: Union[NDArray[float64], str]
    dependency: Union[List[FCDependency], int, str]


class FCMaterialProperties(TypedDict, total=False):
    elasticity: List[FCMaterialProperty] # Упругость и вязкоупругость
    common: List[FCMaterialProperty] # Общие свойства
    thermal: List[FCMaterialProperty] # Температурные свойства
    geomechanic: List[FCMaterialProperty] # Геомеханика
    plasticity: List[FCMaterialProperty] # Пластичность
    hardening: List[FCMaterialProperty] # Упрочнение
    creep: List[FCMaterialProperty] # Позучесть
    preload: List[FCMaterialProperty] # Преднагружение
    strength: List[FCMaterialProperty] # Прочность


class FCMaterial(TypedDict):
    id: int
    name: str
    properties: FCMaterialProperties


class FCConstraint(TypedDict):
    id: int
    name: str
    type: int
    master: NDArray[int32]
    slave: NDArray[int32]
    master_dim: int
    slave_dim: int
    properties: Dict[str, Any]


class FCLoadAxis(TypedDict):
    data: Union[NDArray[float64], str]
    dependency: Union[List[FCDependency], int, str]


class FCLoad(TypedDict):
    apply_to: Union[NDArray[int32], str]
    apply_dim: int
    cs: Optional[int]
    name: str
    type: int
    id: int
    axes: List[FCLoadAxis]


class FCLoads:

    data: Dict[int, FCDict[FCLoad]]

    def __init__(self, data=None):
        self.data = {}

        if data:
            self.decode(data)

    def decode(self, loads_src: List[Dict]):

        for load_src in loads_src:

            axes: List[FCLoadAxis] = []

            for i, dep_type in enumerate(load_src.get("dependency_type", [])):
                if dep_type and 'dep_var_num' in load_src:
                    axes.append({
                        "data": fdecode(load_src['data'][i], dtype('float64')),
                        "dependency": decode_dependency(load_src["dependency_type"][i], load_src['dep_var_num'][i]),
                    })
                else:
                    axes.append({
                        "data": fdecode(load_src['data'][i], dtype('float64')),
                        "dependency": dep_type
                    })

            apply_to = fdecode(load_src['apply_to'], dtype('int32'))
            if type(apply_to) == str:
                apply_dim = 0
            else:
                apply_to_size = load_src['apply_to_size']
                assert apply_to_size != 0 and  len(apply_to)%apply_to_size == 0
                apply_dim = len(apply_to)//apply_to_size

            load: FCLoad = {
                "id": load_src['id'],
                "name": load_src['name'],
                "cs": load_src['cs'] if 'cs' in load_src else 0,
                "apply_to": apply_to,
                "apply_dim": apply_dim,
                "axes": axes,
                "type": load_src['type'],
            }
            
            self.add(load) 


    def encode(self, loads_src):

        for load in self:

            load_src = {
                'id': load['id'],
                'name': load['name'],
                'cs': load['cs'],
                'type': load['type'],
                'apply_to': fencode(load['apply_to']),
                'apply_to_size': len(load['apply_to'])//load['apply_dim'] if load['apply_dim'] else 0,
                'data': [],
                'dependency_type': [],
                'dep_var_num': [],
                'dep_var_size': [],
            }

            for axis in load['axes']:
                load_src['data'].append(fencode(axis['data']))

                const_types, const_dep = encode_dependency(axis["dependency"])

                load_src['dependency_type'].append(const_types)
                load_src['dep_var_num'].append(const_dep)
                load_src['dep_var_size'].append(len(axis['data']) if const_dep else 0)

            loads_src.append(load_src)

    def __len__(self):
        return sum([len(self.data[typeid]) for typeid in self.data])

    def __bool__(self):
        return len(self) > 0

    def __iter__(self):
        for typeid in self.data:
            for elem in self.data[typeid]:
                yield elem

    def add(self, item: FCLoad):

        if item['type'] not in self.data:
            self.data[item['type']] = FCDict()

        self.data[item['type']].add(item)



class FCRestrainAxis(TypedDict):
    data: Union[NDArray[float64], str]
    dependency: Union[List[FCDependency], int, str]
    flag: Union[int, bool]


class FCRestraint(TypedDict):
    apply_to: Union[NDArray[int32], str]
    cs: Optional[int]
    name: str
    id: int
    axes: List[FCRestrainAxis]



class FCSet(TypedDict):
    apply_to: NDArray[int64]
    id: int
    name: str


class FCReciver(TypedDict):
    apply_to: Union[NDArray[int32], str]
    dofs: List[int]
    id: int
    name: str
    type: int


class FCPropertyTable(TypedDict):
    id: int
    type: int
    properties: Dict[str, Any]
    additional_properties: Dict[str, Any]


def decode_dependency(deps_types: Union[List[int], int, str], dep_data) -> Union[List[FCDependency], int, str]:
    if isinstance(deps_types, list):

        return [{
            "type": deps_type,
            "data": fdecode(dep_data[j], dtype(float64))
        } for j, deps_type in enumerate(deps_types)]

    elif isinstance(deps_types, int) or isinstance(deps_types, str):
        return deps_types


def encode_dependency(dependency: Union[List[FCDependency], int, str]):
    if isinstance(dependency, int) or isinstance(dependency, str):
        return dependency, ''
    elif isinstance(dependency, list):
        return [deps['type'] for deps in dependency], [fencode(deps['data']) for deps in dependency]


class FCModel:

    header: FCHeader = {
        "binary": True,
        "description": "Fidesys Case Format",
        "types": {"char": 1, "short_int": 2, "int": 4, "double": 8, },
        "version": 3
    }

    blocks: FCDict[FCBlock]

    coordinate_systems: FCDict[FCCoordinateSystem]

    nodes: FCDict[FCNode]

    elems: FCElems

    materials: FCDict[FCMaterial]

    property_tables: FCDict[FCPropertyTable]

    loads: FCLoads

    restraints: FCDict[FCRestraint]

    contact_constraints: FCDict[FCConstraint]

    coupling_constraints: FCDict[FCConstraint]

    receivers: FCDict[FCReciver]

    nodesets: FCDict[FCSet]
    sidesets: FCDict[FCSet]

    settings = {}

    # periodic_constraints: List[Any] = []
    # initial_sets = []


    def __init__(self, filepath=None):

        self.blocks = FCDict()
        self.coordinate_systems = FCDict()
        self.nodes = FCDict()
        self.elems = FCElems()
        self.materials = FCDict()
        self.property_tables = FCDict()
        self.loads = FCLoads()
        self.restraints = FCDict()
        self.contact_constraints = FCDict()
        self.coupling_constraints = FCDict()
        self.receivers = FCDict()
        self.nodesets = FCDict()
        self.sidesets = FCDict()

        if filepath:
            with open(filepath, "r") as f:
                input_data = json.load(f)

            self.src_data = input_data
            self._decode_header(input_data)
            self._decode_blocks(input_data)
            self._decode_coordinate_systems(input_data)
            self._decode_contact_constraints(input_data)
            self._decode_coupling_constraints(input_data)
            self._decode_mesh(input_data)
            self._decode_settings(input_data)
            self._decode_materials(input_data)
            self._decode_restraints(input_data)
            self._decode_loads(input_data)
            self._decode_receivers(input_data)
            self._decode_property_tables(input_data)
            self._decode_sets(input_data)


    def save(self, filepath):

        output_data = {}

        self._encode_blocks(output_data)
        self._encode_contact_constraints(output_data)
        self._encode_coordinate_systems(output_data)
        self._encode_coupling_constraints(output_data)
        self._encode_header(output_data)
        self._encode_loads(output_data)
        self._encode_materials(output_data)
        self._encode_mesh(output_data)
        self._encode_receivers(output_data)
        self._encode_restraints(output_data)
        self._encode_settings(output_data)
        self._encode_property_tables(output_data)
        self._encode_sets(output_data)

        with open(filepath, "w") as f:
            json.dump(output_data, f, indent=4)


    def _decode_header(self, input_data):
        self.header = input_data.get('header')


    def _encode_header(self, output_data):
        output_data['header'] = self.header


    def _decode_blocks(self, input_data):
        for block in input_data.get('blocks'):
            self.blocks[block['id']] = {
                "id": block['id'],
                "material_id": block['material_id'],
                "property_id": block['property_id'],
                "cs_id": block['cs_id']
            }


    def _encode_blocks(self, output_data):
        if self.blocks:
            output_data['blocks'] = [{
                "id": block['id'],
                "material_id": block['material_id'],
                "property_id": block['property_id'],
                "cs_id": block['cs_id']
            } for block in self.blocks]


    def _decode_coordinate_systems(self, input_data):

        for cs in input_data.get('coordinate_systems'):
            self.coordinate_systems[cs['id']] = {
                'dir1': decode(cs['dir1'], dtype(float64)),
                'dir2': decode(cs['dir2'], dtype(float64)),
                'origin': decode(cs['origin'], dtype(float64)),
                "id": cs['id'],
                "name": cs['name'],
                "type": cs['type']
            }

    def _encode_coordinate_systems(self, output_data):

        if self.coordinate_systems:
            output_data['coordinate_systems'] = [{
                'dir1': encode(cs['dir1']),
                'dir2': encode(cs['dir2']),
                'origin': encode(cs['origin']),
                "id": cs['id'],
                "name": cs['name'],
                "type": cs['type']
            } for cs in self.coordinate_systems]


    def _decode_contact_constraints(self, input_data):

        for cc_src in input_data.get('contact_constraints', []):
            master = decode(cc_src['master'], dtype(int32))
            slave = decode(cc_src['slave'], dtype(int32))

            self.contact_constraints[cc_src['id']] = {
                'id': cc_src['id'],
                'name': cc_src['name'],
                'type': cc_src['type'],
                'master': master,
                'master_dim': len(master)//cc_src['master_size'] if cc_src['master_size'] else 0,
                'slave': slave,
                'slave_dim': len(slave)//cc_src['slave_size'] if cc_src['slave_size'] else 0,
                'properties': {
                    key:cc_src[key] for key in cc_src
                    if key not in [
                        'id','name','type',
                        'master','master_size',
                        'slave','slave_size',
                    ]}
            }


    def _encode_contact_constraints(self, output_data):
        if self.contact_constraints:
            output_data['contact_constraints'] = []

            for cc in self.contact_constraints:
                cc_src = {
                    'id': cc['id'],
                    'type': cc['type'],
                    'name': cc['name'],
                    'master': encode(cc['master']),
                    'master_size': len(cc['master'])//cc['master_dim'] if cc['master_dim'] else 0,
                    'slave': encode(cc['slave']),
                    'slave_size': len(cc['slave'])//cc['slave_dim'] if cc['slave_dim'] else 0,
                }
                for key in cc['properties']:
                    cc_src[key] = cc['properties'][key]

                output_data['contact_constraints'].append(cc_src)


    def _decode_coupling_constraints(self, input_data):

        for cc_src in input_data.get('coupling_constraints', []):
            master = decode(cc_src['master'], dtype(int32))
            slave = decode(cc_src['slave'], dtype(int32))

            self.coupling_constraints[cc_src['id']] = {
                'id': cc_src['id'],
                'name': cc_src['name'],
                'type': cc_src['type'],
                'master': master,
                'master_dim': len(master)//cc_src['master_size'] if cc_src['master_size'] else 0,
                'slave': slave,
                'slave_dim': len(slave)//cc_src['slave_size'] if cc_src['slave_size'] else 0,
                'properties': {
                    key:cc_src[key] for key in cc_src
                    if key not in [
                        'id','name','type',
                        'master','master_size',
                        'slave','slave_size',
                    ]}
            }


    def _encode_coupling_constraints(self, output_data):
        if self.coupling_constraints:
            output_data['coupling_constraints'] = []

            for cc in self.coupling_constraints:
                cc_src = {
                    'id': cc['id'],
                    'type': cc['type'],
                    'name': cc['name'],
                    'master': encode(cc['master']),
                    'master_size': len(cc['master'])//cc['master_dim'] if cc['master_dim'] else 0,
                    'slave': encode(cc['slave']),
                    'slave_size': len(cc['slave'])//cc['slave_dim'] if cc['slave_dim'] else 0,
                }
                for key in cc['properties']:
                    cc_src[key] = cc['properties'][key]

                output_data['coupling_constraints'].append(cc_src)


    def _decode_sets(self, src_data):

        if 'sets' in src_data:

            for nodeset_src in src_data['sets'].get('nodesets', []):
                self.nodesets[nodeset_src['id']] = {
                    'id': nodeset_src['id'],
                    'name': nodeset_src['name'],
                    'apply_to': decode(nodeset_src['apply_to'], dtype(int32))
                }

            for sideset_src in src_data['sets'].get('sidesets', []):
                self.sidesets[sideset_src['id']] = {
                    'id': sideset_src['id'],
                    'name': sideset_src['name'],
                    'apply_to': decode(sideset_src['apply_to'], dtype(int32)),
                }



    def _encode_sets(self,  src_data):

        if not (self.nodesets or self.sidesets):
            return

        src_data['sets'] = {}

        if self.nodesets:
            src_data['sets']['nodesets'] = [{
                'id': nodeset['id'],
                'name': nodeset['name'],
                'apply_to': encode(nodeset['apply_to']),
                'apply_to_size': len(nodeset['apply_to']),
            } for nodeset in self.nodesets]

        if self.sidesets:
            src_data['sets']['sidesets'] = [{
                'id': sideset['id'],
                'name': sideset['name'],
                'apply_to': encode(sideset['apply_to']),
                'apply_to_size': len(sideset['apply_to']),
            } for sideset in self.sidesets]


    def _decode_mesh(self, src_data):

        node_ids = decode(src_data['mesh']['nids'])
        node_coords = decode(src_data['mesh']['nodes'], dtype('float64')).reshape(-1, 3)

        for i, nid in enumerate(node_ids):

            self.nodes[nid] = {
                'id': nid,
                'xyz':node_coords[i]
            }

        self.elems.decode(src_data['mesh'])


    def _encode_mesh(self, src_data):

        nodes_count = len(self.nodes)
        node_ids = np.zeros(nodes_count, np.int32)
        node_xyzs = np.zeros((nodes_count,3), np.float64)

        for i, node in enumerate(self.nodes):
            node_ids[i] = node['id']
            node_xyzs[i] = node['xyz']

        src_data['mesh'] = {
            "nodes_count": nodes_count,
            "nids": encode(node_ids),
            "nodes": encode(node_xyzs),
            **self.elems.encode()
        }


    def _decode_settings(self, src_data):
        self.settings = src_data.get('settings')


    def _encode_settings(self, src_data):
        settings = self.settings
        src_data['settings'] = settings


    def _decode_property_tables(self, src_data):
        for property_table in src_data.get('property_tables', []):
            self.property_tables[property_table['id']] = {
                'id': property_table['id'],
                'type': property_table['type'],
                'properties': property_table['properties'],
                'additional_properties': {key:property_table[key] for key in property_table if key not in ['id', 'type', 'properties']},
            }


    def _encode_property_tables(self, src_data):
        if self.property_tables:
            src_data['property_tables'] = [{
                'id': value['id'],
                'type': value['type'],
                'properties': value['properties'],
                **value['additional_properties']
            } for value in self.property_tables]


    def _decode_materials(self, src_data):

        for material_src in src_data.get('materials', []):

            properties: FCMaterialProperties = {}

            for property_name in material_src:
                properties_src = material_src[property_name]

                if not isinstance(properties_src, list):
                    continue

                properties[property_name] = []

                for property_src in properties_src:
                    for i, constants in enumerate(property_src["constants"]):

                        property: FCMaterialProperty = {
                            "name": property_src["const_names"][i],
                            "data": decode(constants, dtype(float64)),
                            "type": property_src["type"],
                            "dependency": decode_dependency(
                                property_src["const_types"][i],
                                property_src["const_dep"][i]
                            )
                        }

                        properties[property_name].append(property)

            self.materials[material_src['id']] = {
                "id": material_src['id'],
                "name": material_src['name'],
                "properties": properties
            }


    def _encode_materials(self, src_data):

        if self.materials:

            src_data['materials'] = []

            for material in self.materials:

                material_src = {
                    "id": material['id'],
                    "name": material['name'],
                }

                for property_name in material["properties"]:

                    material_src[property_name] = []

                    for material_property in material["properties"][property_name]:

                        const_types, const_dep = encode_dependency(material_property["dependency"])

                        material_src[property_name].append({
                            "const_dep": [const_dep],
                            "const_dep_size": [len(material_property["data"])],
                            "const_names": [material_property["name"]],
                            "const_types": [const_types],
                            "constants": [fencode(material_property["data"])],
                            "type": material_property["type"]
                        })

                src_data['materials'].append(material_src)


    def _decode_restraints(self, src_data):

        for restraint_src in src_data.get('restraints', []):

            axes: List[FCRestrainAxis] = []

            for i, dep_type in enumerate(restraint_src.get("dependency_type", [])):
                if 'dep_var_num' in restraint_src:
                    axes.append({
                        "data": fdecode(restraint_src['data'][i], dtype('float64')),
                        "dependency": decode_dependency(dep_type, restraint_src['dep_var_num'][i]),
                        "flag": restraint_src['flag'][i],
                    })

            apply_to = fdecode(restraint_src['apply_to'], dtype('int32'))

            if type(apply_to) != str:
                assert len(apply_to) == restraint_src['apply_to_size']

            self.restraints[restraint_src['id']] = {
                "id": restraint_src['id'],
                "name": restraint_src['name'],
                "cs": restraint_src['cs'] if 'cs' in restraint_src else 0,
                "apply_to": apply_to,
                "axes": axes,
            }


    def _encode_restraints(self, src_data):

        if self.restraints:

            src_data['restraints'] = []

            for restraint in self.restraints:

                apply_to = fencode(restraint['apply_to'])

                restraint_src = {
                    'id': restraint['id'],
                    'name': restraint['name'],
                    'cs': restraint['cs'],
                    'apply_to': apply_to,
                    'apply_to_size': len(restraint['apply_to']) if type(restraint['apply_to']) != str else 0,
                    'data': [],
                    'flag': [],
                    'dependency_type': [],
                    'dep_var_num': [],
                    'dep_var_size': [],
                }


                for axis in restraint['axes']:
                    if type(axis) == dict:
                        restraint_src['data'].append(fencode(axis['data']))
                        restraint_src['flag'].append(axis['flag'])

                        const_types, const_dep = encode_dependency(axis["dependency"])

                        restraint_src['dependency_type'].append(const_types)
                        restraint_src['dep_var_num'].append(const_dep)
                        restraint_src['dep_var_size'].append(len(axis['data']) if const_dep else 0)
                    else:
                        restraint_src['data'].append(fencode(axis['data']))
                        restraint_src['dependency_type'].append(0)
                        restraint_src['dep_var_num'].append("")
                        restraint_src['dep_var_size'].append(0)
                        restraint_src['flag'].append(15)

                src_data['restraints'].append(restraint_src)


    def _decode_loads(self, src_data):
        self.loads.decode(src_data.get('loads', []))


    def _encode_loads(self, src_data):

        if self.loads:
            src_data['loads'] = []
            self.loads.encode(src_data['loads'])


    def _decode_receivers(self, src_data):

        for r in src_data.get('receivers', []):
            receiver: FCReciver = {
                'apply_to': fdecode(r['apply_to']),
                'dofs': r['dofs'],
                "id": r['id'],
                "name": r['name'],
                "type": r['type']
            }
            assert len(receiver['apply_to']) == r['apply_to_size']

            self.receivers[r['id']] = receiver


    def _encode_receivers(self, src_data):

        if self.receivers:
            src_data['receivers'] = [{
                'apply_to': fencode(r['apply_to']),
                'apply_to_size': len(r['apply_to']),
                'dofs': r['dofs'],
                "id": r['id'],
                "name": r['name'],
                "type": r['type']
            } for r in self.receivers]



if __name__ == '__main__':
    name = "model5_rec_fix_bc"
    datapath = "./data/"

    inputpath = os.path.join(datapath, f"{name}.fc")
    outputpath = os.path.join(datapath, f"{name}_new.fc")

    fc_model = FCModel(inputpath)

    fc_model.save(outputpath)
