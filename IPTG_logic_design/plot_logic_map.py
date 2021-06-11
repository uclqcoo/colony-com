import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'IPTG_characterisation'))


from growth_fitting import gompertz
from fitting_with_gompertz import *
import numpy as np
import matplotlib.pyplot as plt
from fitting import plot_IPTG
import itertools
import time
import copy
from matplotlib import cm
from matplotlib import colors
import math
import scipy.ndimage as sn
from design_logic_gates import run_experiment

def point_diffusion(r,t):
    return np.exp(-r**2/(4*D*t))/(4*D*np.pi*t)*C0

if __name__ == '__main__':
    params = [2.14419831, -0.02864081, -1.46149617, 1.86331062] # small concs
    n_inputs = 2
    n_rows = 78
    n_cols = 78
    active_thr = 0.1
    inactive_thr = 0.1

    conc = 2.5

    logic_map = np.zeros((n_rows, n_cols)) + -1

    print(np.sum(logic_map))
    coords = np.array([[39, 30], [39, 48]])#AND
    coords = np.array([[24,  8], [19,  7]]) #AND over time

    coords = np.array([[35, 24], [35, 46]]) #field experiment

    for i in range(2**2**n_inputs):
        logic_gate = list(map(int, list(str(bin(i))[2:].zfill(4))))
        print(logic_gate)



        all_outputs = list(map(np.array, list(itertools.product([0, 1], repeat=n_inputs))))


        scores = []
        # 0 IPTG
        # 1 Nutrients
        # 2 Receiver
        # 3 GFP



        logic_area = np.ones((n_rows, n_cols), dtype=np.int32) # start off as all ones and eliminate

        for j in range( 2**n_inputs):
            sim_ivp = run_experiment(params, coords[all_outputs[j] == 1], conc, n_rows, n_cols)

            GFP = sim_ivp[3]
            end_GFP = GFP[:,:,-1]



            #f, ax = plt.subplots(1,1)


            #im1 = ax.imshow(sim_ivp[0,:,:,-1], interpolation="none", cmap=cm.viridis, vmin=0)


            #f.colorbar(im1)



            #active = end_GFP > active_thr
            #inactive = (end_GFP < inactive_thr)*-1
            #activation_map = active + inactive

            if logic_gate[j] == 0:
                logic_area[end_GFP > inactive_thr] = 0
            elif logic_gate[j] == 1:
                logic_area[end_GFP < active_thr] = 0

        logic_map[logic_area==1] = i
        print(np.sum(logic_area), np.sum(logic_map))

    print(logic_map)
    print(np.max(logic_map))
    print(np.min(logic_map))
    f, ax = plt.subplots(1,1)
    cmap = cm.viridis
    bounds = list(np.arange(-0.5, 16.5, 1))
    print(bounds)
    norm = colors.BoundaryNorm(bounds, cmap.N)
    im1 = ax.imshow(logic_map, interpolation="none", cmap=cm.get_cmap('Set2'))

    f.colorbar(im1)
    plt.show()








