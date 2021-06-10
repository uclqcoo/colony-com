
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

import math
import scipy.ndimage as sn

def point_diffusion(r,t):
    return np.exp(-r**2/(4*D*t))/(4*D*np.pi*t)*C0

def run_experiment(params, IPTG_coords, conc, n_rows, n_cols, plot = False):

    n0 = 25
    w = 1 #dx, dy
    receiver_radius = 2

    U = np.zeros([4, n_rows, n_cols])

    shape = U.shape
    size = U.size

    U[1] = n0 #set nutrients

    #U[1] = 43400000#nM

    #IPTG
    vol = 1e-6 #  litres

    IPTG_conc=conc
    amount = vol*conc # milli moles

    #print('initial amoun of IPTG: ', amount)
    agar_thickness = 3.12 #mm


    init_conc = amount/(w**2*agar_thickness) # conc if all IPTG is put into one grid point mmol/mm^3
    init_conc *= 1e6 # mM


    #print('initial IPTG conc:', init_conc)

    rows = IPTG_coords[:,0]
    cols = IPTG_coords[:,1]
    U[0][rows, cols] = init_conc

    dist = 4.5 #mm
    centre = 0

    receiver_pos = [[centre - i * dist, centre] for i in range(1, 4)]

    all_vertex_numbers = np.arange(n_rows * n_cols).reshape(-1, 1)  # reshpae to colum vecto
    all_vertex_coordinates = get_vertex_coordinates(all_vertex_numbers, n_rows, n_cols)

    receiver_coordinates = all_vertex_coordinates



    t_final = 20*60  # mins
    #t_final = 20 * 47  # mins
    dt = .1
    t_points = int(t_final / dt)

    t = np.arange(0, t_final, dt)
    U_init = U.flatten()
    start_time = time.time()

    func = lambda t, U: model_IPTG(t, U, shape, params, w, receiver_coordinates)

    sim_ivp = solve_ivp(func, [0, t_final], U_init,
                        t_eval=t)

    sim_ivp = sim_ivp.y.reshape(4, n_rows, n_cols, t_points)

    if os.path.isdir(os.getcwd() + "/IPTG_inf/output_cross") is False:
        print("ciao")
        os.makedirs(os.getcwd() + "/IPTG_inf/output_cross")


    if plot:
        with PdfPages("IPTG_inf/output_cross/simulation-cross-" + time.strftime("%H%M%S") + ".pdf") as pdf:
            print('sim ivp:', sim_ivp.shape)

            for i in np.arange(0, t_points+1, int(120/dt)):

                t = int(i / 600)
                f1 = plot_IPTG(sim_ivp[:, :, :, i], str(IPTG_conc) + ': ' + str(t) + " hours" )
                plt.title(str(IPTG_conc))
                pdf.savefig()
                # plt.show()
                plt.close()

        end_time = time.time()
        print(end_time - start_time)

        print(sim_ivp.shape)

    return sim_ivp

def get_distance(coords):
    a = coords[0][0] - coords[1][0]
    b = coords[0][1] - coords[1][1]

    return (a**2 + b**2)**0.5

def get_fitness(coords):
    logic_area = np.ones((35, 35))  # start off as all ones and eliminate
    for i in range(2 ** n_inputs):

        sim_ivp = run_experiment(params, coords[all_outputs[i] == 1], conc, n_rows, n_cols)

        GFP = sim_ivp[3]
        end_GFP = GFP[:, :, -1]

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
    return score

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

        if score > max_score:
            max_t = t
            max_score = score

    return max_score, max_t


if __name__ == '__main__':
    params = [3.62992409, -0.99913788, -2.67173002, -0.22840798] # large concs
    params = [2.14419831, -0.02864081, -1.46149617, 1.86331062] # small concs
    n_inputs = 2
    min_distance = 4.5
    logic_gate = [0, 0, 0, 1]
    active_thr = 0.35
    inactive_thr = 0.25
    activation_ratio = 2
    pop_size = 100
    n_gens = 100
    conc = 2.5
    n_rows = n_cols = 35

    IPTG_coords = np.random.randint(0, n_rows, size=(pop_size,n_inputs, 2))
    IPTG_coords = []

    t = time.time()
    while len(IPTG_coords) < pop_size:
        coords = np.random.randint(0, n_rows, size=(n_inputs, 2))
        #print(coords, get_distance(coords))
        if get_distance(coords) >= min_distance:
            IPTG_coords.append(coords)
    print('init:', time.time() - t)

    IPTG_coords = np.array(IPTG_coords)
    all_outputs = list(map(np.array, list(itertools.product([0, 1], repeat=n_inputs))))


    simulated = np.zeros((pop_size)) # keep track of grids that have been simulated so we dont do them again
    scores = np.zeros((pop_size))
    for gen in range(n_gens):

        # 0 IPTG
        # 1 Nutrients
        # 2 Receiver
        # 3 GFP

        t = time.time()
        for j in range(pop_size):



            #print(coords, get_distance(coords))
            #coords = np.array([[15,32],[33, 13]]) # AND, score = 6
            #coords = np.array([[20,14],[1, 32]]) # AND, score = 6
            #coords = np.array([[27, 25], [27, 26]])
            #coords = np.array([[23, 28], [4, 23]]) #AND , score = 35
            #coords = np.array([[24  8],[19  7]]) #AND over time

            if not simulated[j]:

                score, _ = max_fitness_over_t(IPTG_coords[j])

                simulated[j] = 1
                scores[j] = score

        print('simulation:', time.time() - t)

        ind = np.argsort(scores)[::-1]

        scores = scores[ind]
        IPTG_coords = IPTG_coords[ind]
        simulated = simulated[ind]
        new_IPTG_coords = copy.deepcopy(IPTG_coords)


        #mutate bottom half
        t = time.time()
        for i in range(int(pop_size/2), int(pop_size*0.9)):

            simulated[i] = 0
            new_IPTG_coords[i] += np.random.randint(-3, 3, size = ( n_inputs, 2))
            new_IPTG_coords[i] = np.clip(new_IPTG_coords[i], 0, 34)

            while get_distance(new_IPTG_coords[i]) < min_distance:
                new_IPTG_coords[i] += np.random.randint(-3, 3, size=(n_inputs, 2))
                new_IPTG_coords[i] = np.clip(new_IPTG_coords[i], 0, 34)


        # add random grids for bottom 10%
        for i in range(int(pop_size*0.9), pop_size):

            simulated[i] = 0
            new_IPTG_coords[i] = np.random.randint(0, 35, size=( n_inputs, 2))
            new_IPTG_coords[i] = np.clip(new_IPTG_coords[i], 0, 34)

            while get_distance(new_IPTG_coords[i]) < min_distance:
                new_IPTG_coords[i] = np.random.randint(0, 35, size=( n_inputs, 2))
                new_IPTG_coords[i] = np.clip(new_IPTG_coords[i], 0, 34)


        IPTG_coords = copy.deepcopy(new_IPTG_coords)
        print('mutate:', time.time() -t)
        print("gen : " + str(gen) + ", av score: ", np.mean(scores), " top score: ", np.max(scores))
        print(IPTG_coords[0])
        print(scores)


