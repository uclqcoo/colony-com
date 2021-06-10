


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
from matplotlib import colors
import math
import scipy.ndimage as sn

def point_diffusion(r,t):
    return np.exp(-r**2/(4*D*t))/(4*D*np.pi*t)*C0

def run_experiment(params, IPTG_coords, concs, n_rows, n_cols, plot = False):

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

    for i, conc in enumerate(concs):
        IPTG_conc=conc
        amount = vol*conc # milli moles

        #print('initial amoun of IPTG: ', amount)
        agar_thickness = 3.12 #mm


        init_conc = amount/(w**2*agar_thickness) # conc if all IPTG is put into one grid point mmol/mm^3
        init_conc *= 1e6 # mM


        #print('initial IPTG conc:', init_conc)

        rows = IPTG_coords[:,0]
        cols = IPTG_coords[:,1]

        U[0][rows[i], cols[i]] = init_conc

    dist = 4.5 #mm
    centre = 0

    receiver_pos = np.array([[21, 11]])
    receiver_coordinates = get_node_coordinates(receiver_pos, receiver_radius, n_rows, n_cols, w)

    t_final = 13*60  # mins
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


    return sim_ivp


if __name__ == '__main__':
    params = [ 2.00434884, -0.00518 ,   -1.42864773,  1.99791128]
    n_inputs = 2

    logic_gate = [0, 0, 0, 1]
    n_rows = n_cols = 35
    active_thr = 0.2
    inactive_thr = 0.2
    pop_size = 100
    n_gens = 100

    IPTG_coords = np.random.randint(0, n_rows, size=(pop_size,n_inputs, 2))
    all_outputs = list(map(np.array, list(itertools.product([0, 1], repeat=n_inputs))))


    scores = []
    # 0 IPTG
    # 1 Nutrients
    # 2 Receiver
    # 3 GFP



    logic_area = np.ones((n_rows, n_cols)) # start off as all ones and eliminate



    coords = np.array([[15,32],[33, 13]])
    coords = np.array([[20,14],[1, 32]])
    coords = np.array([[24, 8], [19, 7]])




    all_data= []

    x = y = np.arange(0, 2.6,  0.1)

    for IPTG_1 in x:
        data = []


        for IPTG_2 in y:


            sim_ivp = run_experiment(params, coords, [IPTG_1, IPTG_2], n_rows, n_cols)

            GFP = sim_ivp[3]
            end_GFP = GFP[:,:,-1]
            data.append(np.sum(end_GFP)/9)

            if IPTG_1 == 2.5 and IPTG_2 == 0:
                print(np.sum(end_GFP)/9)
            elif IPTG_1 == 0 and IPTG_2 == 2.5:
                print(np.sum(end_GFP)/9)
            elif IPTG_1 == 2.5 and IPTG_2 == 2.5:
                print(np.sum(end_GFP)/9)


        all_data.append(data)

    all_data = np.array(all_data)


    threshold = 0.3
    classification = all_data > threshold

    fig = plt.figure()

    ax = fig.add_subplot(projection='3d')
    X, Y = np.meshgrid(x,y)
    ax.set_xlabel('[IPTG] input 1')
    ax.set_ylabel('[IPTG] input 2')
    ax.set_zlabel('Mean output flourescence per pixel')
    ax.plot_wireframe(X, Y, all_data)

    plt.figure()
    plt.imshow(np.flip(all_data, axis=0))
    plt.xlabel('[IPTG] input 1')
    plt.ylabel('[IPTG] input 2')
    plt.xticks(range(len(x))[::2], np.around(x[::2], decimals=1))
    plt.yticks(range(len(y))[::2], np.around(y[::-2], decimals=1))
    cbar = plt.colorbar()
    cbar.set_label('mean pixel flourescence', labelpad = 40, rotation = 270)

    plt.figure()
    cmap = colors.ListedColormap(['red', 'green'])
    plt.xlabel('[IPTG] input 1')
    plt.xlabel('[IPTG] input 2')
    plt.xticks(range(len(x))[::2], np.around(x[::2], decimals=1))
    plt.yticks(range(len(y))[::2], np.around(y[::-2], decimals=1))
    plt.imshow(np.flip(classification, axis=0), cmap = cmap)
    plt.show()



