#-*- coding:utf-8 -*-
from math import pi, sin, cos

def Lame(param):

    rMin, rMax, resolution = float(param['rMin']), float(param['rMax']), param['resolution']

    nodes = []
    elems = []

    radden = int(2*resolution/pi*(rMax-rMin)/rMin) + 1

    delta = (rMax-rMin)/radden

    bounds = [
        {
        "nodes":[],
        'type':'neumann',
        'value':param['P1']
        },
        {
        "nodes":[],
        'type':'dirichlet',
        'fix':1,
        'value':0,
        },
        {
        "nodes":[],
        'type':'neumann',
        'value':param['P2']
        },
        {
        "nodes":[],
        'type':'dirichlet',
        'fix':0,
        'value':0,
        },
    ]

    material = {
        "E": param['E'], 
        "Nu": param['Nu']
    }

    #генерируем узлы сетки
    i = 0
    for r in range(radden+1):
        for f in range(resolution+r+1):
            coord = shpereCoords(rMin+delta*r, float(f)/(resolution+r))
            nodes.append(coord)

            if r==0:
                bounds[0]["nodes"].append(i)
            if f==0:
                bounds[1]["nodes"].append(i)
            if r==radden:
                bounds[2]["nodes"].append(i)
            if f==(resolution+r):
                bounds[3]["nodes"].append(i)

            i += 1

    bounds[0]["nodes"].reverse()

    #генерируем треугольные элементы сетки
    for r in range(radden+1):
        for f in range(resolution+r):
            a = (2*resolution+r+1)*r//2 + f
            b = (2*resolution+r+1)*r//2 + f + 1
            c1 = (2*resolution+r+2)*(r+1)//2 + f + 1
            c2 = (2*resolution+r)*(r-1)//2 + f
            if r != 0:
                elems.append([a,b,c2])
            if r != radden:
                elems.append([a,c1,b])


    return {
        'nodes': nodes,
        'elems': elems,
        'bounds': bounds,
        'material': material
    }



def shpereCoords(r, fi):
    fi = fi*pi/2
    return [r*cos(fi), r*sin(fi)]




def json2inp(task):
    file = """*HEADING
cubit(task.inp): 11/21/2018: 13
version: 16.5.3
**
********************************** P A R T S **********************************
*PART, NAME=Part-Default
**
********************************** N O D E S **********************************
*NODE, NSET=ALLNODES
"""
    for i, node in enumerate(task['nodes']):
        file += "    {},    {},    {},    {}\n".format(i, format(node[0], '.6e'), format(node[1], '.6e'), format(0, '.6e'))

    file += """**
********************************** E L E M E N T S ****************************
*ELEMENT, TYPE=STRI3, ELSET=EB1
"""
    for i, elem in enumerate(task['elems']):
        file += "{},    {},    {},    {}\n".format(i, elem[0], elem[1], elem[2])

    file += """**
********************************** P R O P E R T I E S ************************
*SHELL SECTION, ELSET=EB1, SECTION INTEGRATION=SIMPSON, MATERIAL=Default-Steel
1.000000e+00
**
*END PART
**
**
**
********************************** E N D   P A R T S **********************************
**
**
********************************** A S S E M B L Y ************************************
**
*ASSEMBLY, NAME=ASSEMBLY1
**
*INSTANCE, NAME=Part-Default_1, PART=Part-Default
*END INSTANCE
**
*END ASSEMBLY
**
**
**
*MATERIAL, NAME = Default-Steel
*ELASTIC, TYPE=ISOTROPIC
2.068000e+05, 2.900000e-01
*DENSITY
7.000000e-06
*CONDUCTIVITY,TYPE=ISO
4.500000e-02
*SPECIFIC HEAT
5.000000e+02
**
**
************************************** H I S T O R Y *************************************
**
*PREPRINT
**
**************************************** S T E P 1 ***************************************
*STEP,INC=100,NAME=Default Set
**
*STATIC
1, 1, 1e-05, 1
**
**
**
**
*END STEP
"""

    return file