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

import math
import scipy.ndimage as sn
from design_logic_gates import run_experiment

def point_diffusion(r,t):
    return np.exp(-r**2/(4*D*t))/(4*D*np.pi*t)*C0


def max_fitness_over_t(coords):

    all_sims = []
    for i in range(2 ** n_inputs):

        sim_ivp = run_experiment(params, coords[all_outputs[i] == 1], conc, n_rows, n_cols)
        all_sims.append(sim_ivp)

    max_score = 0
    max_t = 0

    n_t = sim_ivp.shape[3]
    for t in list(range(int(n_t/20), int(n_t), int(n_t/20))) + [n_t -1]:
        logic_area = np.ones((35, 35))  # start off as all ones and eliminate
        for i in range(2 ** n_inputs):

            sim_ivp = all_sims[i]
            GFP = sim_ivp[3]
            end_GFP = GFP[:, :, t]

            #plt.figure()
            #plt.imshow(end_GFP)
            # plt.imshow(sim_ivp[0, :, :, -1])

            # print('max: ', np.max(end_GFP))
            # print('mean: ', np.mean(end_GFP))

            # active = end_GFP > active_thr
            # inactive = (end_GFP < inactive_thr)*-1
            # activation_map = active + inactive

            if logic_gate[i] == 0:
                logic_area[end_GFP > inactive_thr] = 0
            elif logic_gate[i] == 1:
                logic_area[end_GFP < active_thr] = 0
        #plt.figure()
        #plt.imshow(logic_area)
        #plt.show()
        labels, n = sn.label(logic_area)
        sizes = sn.sum(logic_area, labels, range(1, n + 1))

        # score = np.sum(logic_area)

        try:
            score = np.max(sizes)
        except:
            score = 0
        print(score)
        if score > max_score:
            max_t = t
            max_score = score

    return max_score, max_t

if __name__ == '__main__':
    params = [2.14419831, -0.02864081, -1.46149617, 1.86331062] # small concs
    n_inputs = 2

    logic_gate = [0, 0, 0, 1]
    n_rows = 35
    n_cols = 35
    active_thr = 0.35
    inactive_thr = 0.25
    pop_size = 100
    n_gens = 100
    conc = 2.5

    IPTG_coords = np.random.randint(0, n_rows, size=(pop_size,n_inputs, 2))
    all_outputs = list(map(np.array, list(itertools.product([0, 1], repeat=n_inputs))))


    scores = []
    # 0 IPTG
    # 1 Nutrients
    # 2 Receiver
    # 3 GFP



    logic_area = np.ones((n_rows, n_cols)) # start off as all ones and eliminate


    #coords = np.array([[20, 25], [22, 23]])
    coords = np.array([[15,32],[33, 13]])
    #coords = np.array([[20,14],[1, 32]])
    coords = np.array([[24,  8],[19,  7]])
    coords = np.array([[11, 11], [20, 20]])

    score, t = max_fitness_over_t(coords)
    print('best time  (hours):', t/600)

    for i in range( 2**n_inputs):

        sim_ivp = run_experiment(params, coords[all_outputs[i] == 1], conc, n_rows, n_cols)
        print(sim_ivp.shape[3])
        print(sim_ivp.shape[3])
        GFP = sim_ivp[3]
        end_GFP = GFP[:,:,t]



        f, ax = plt.subplots(1,1)


        im1 = ax.imshow(sim_ivp[0,:,:,-1], interpolation="none", cmap=cm.viridis, vmin=0)


        f.colorbar(im1)
        print(end_GFP.shape)


        #active = end_GFP > active_thr
        #inactive = (end_GFP < inactive_thr)*-1
        #activation_map = active + inactive

        if logic_gate[i] == 0:
            logic_area[end_GFP > inactive_thr] = 0
        elif logic_gate[i] == 1:
            logic_area[end_GFP < active_thr] = 0



    labels, n = sn.label(logic_area)
    sizes = sn.sum(logic_area, labels, range(1, n+1))

    #score = np.sum(logic_area)

    plt.figure()
    plt.imshow(logic_area)
    plt.figure()
    plt.imshow(sim_ivp[0, :, :, -1])
    plt.imshow(sim_ivp[-1, :, :, -1])

    plt.show()

    try:
        score = np.max(sizes)

    except:
        score = 0


    scores.append(score)






