
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'parameter_estimation'), 'global_optimsiation'))


from functionsDiffusion import *
import numpy as np
import math
import time
from load_spot_data import *

from particle_swarm_optimiser import *
from matplotlib.backends.backend_pdf import PdfPages

import pyswarms as ps

from growth_fitting import gompertz

n0 = 500
init_cells = 0.5

# fixed global parameters from Doong et. al. 2017
Dc        = 0.0001     # diffusion cefficient of cells mm^2/min
rc        = 6e-4 # max growth rate #1/min
#x_g       = .1         # max production of GFP
rho_n     = 3          # yield coefficient 1/min
lambda_n  = 2.0
K_n       = 80 # arbitrary
D_n = 0.03 #diffusion coeff of nutrients mm^2/min


lambda_a  = 2.3        # hill coeff IPTG
K_a       = 40e-6         # sat constant for IPTG mM
D         = 0.03       # IPTG diffusion coefficitient mm^2/min


def hill_IPTG(conc, n, kd, min, max):
    # get rid of the very small negatvie values
    conc[conc<0] = 0
    h = (min + (max-min)*(conc**n/(kd**n + conc**n)))
    return h

def hill_BP(conc, na, ka, nr, kr, min, max):
    conc[conc < 0] = 0
    act = conc**na/(ka**na + conc**na)
    rep = kr**nr/(kr**nr + conc**nr)

    #print()
    return min + (max-min)*act*rep

#gompertz_ps = [1.34750462e-01, 1.90257947e-04, 3.33841052e+02] #threshold

gompertz_ps = [1.24927088e-01, 1.79595968e-04, 3.48051876e+02] #bandpass

cell_conc = lambda t: gompertz(t,*gompertz_ps)

def model_IPTG(t, U_flat, shape, params, w, receiver_coordinates):

    #lambda_a, K_a, D = params
    #lambda_a, K_a, D, max = [10**p for p in params] #convert from log scale
    #lambda_a, K_a, D, max,n0 = [10**p for p in params] #convert from log scale
    #lambda_a, K_a, D, max,n0 = params
    #lambda_a, K_a, D, max = params

    D = params[0]

    K_a = 10**K_a
    max = 10**max
    D = 10**D

    #lambda_a, K_a, D, K_n, lambda_n, Dc, rc, D_n, rho_n, max = params

    U_grid = U_flat.reshape(shape)
    # 0 IPTG
    # 1 Nutrients
    # 2 Receiver
    # 3 GFP

    # set reciever conc according to the gompertz model
    rows = receiver_coordinates[:, 0]
    cols = receiver_coordinates[:, 1]

    U_grid[2][rows, cols] = cell_conc(t)

    N = hill(U_grid[1], K_n, lambda_n)
    IPTG_ficks = ficks(U_grid[0], w)
    n_ficks = ficks(U_grid[1], w)
    R_ficks = ficks(U_grid[2], w)


    IPTG = D * IPTG_ficks
    n = D_n * n_ficks - rho_n*N*U_grid[2]

    gfp =  N *max* hill(U_grid[0], lambda_a, K_a) * U_grid[2]


    return (np.concatenate((IPTG.flatten(),
                            n.flatten(),
                            np.zeros(gfp.shape).flatten(),
                            gfp.flatten())))

def model_IPTG_BP(t, U_flat, shape, params, w, receiver_coordinates):

    na, ka, nr, kr, D, min, max = params
    #D,min, max = params

    #na, ka, nr, kr= [1.58646428, -0.6564445, 7.99974394, -2.14317698] #agar
    #na, ka, nr, kr= [2.60754348, -1.8366539, 4.75478292, -2.16703204] #liquid




    ka = 10**ka
    kr = 10**kr
    min = 10**min
    max = 10**max
    D = 10**D



    U_grid = U_flat.reshape(shape)
    # 0 IPTG
    # 1 Nutrients
    # 2 Receiver
    # 3 GFP

    # set reciever conc according to the gompertz model
    rows = receiver_coordinates[:, 0]
    cols = receiver_coordinates[:, 1]

    U_grid[2][rows, cols] = cell_conc(t)

    N = hill(U_grid[1], K_n, lambda_n)
    IPTG_ficks = ficks(U_grid[0], w)
    n_ficks = ficks(U_grid[1], w)
    R_ficks = ficks(U_grid[2], w)


    IPTG = D * IPTG_ficks

    n = D_n * n_ficks - rho_n*N*U_grid[2]

    gfp =  N*hill_BP(U_grid[0], na, ka, nr, kr, min, max) * U_grid[2]


    return (np.concatenate((IPTG.flatten(),
                            n.flatten(),
                            np.zeros(gfp.shape).flatten(),
                            gfp.flatten())))

def plot_IPTG(sim, title=""):
    f, ax = plt.subplots(2, 2, sharex=True, sharey=False, figsize=(15, 15))

    f.suptitle(title, fontsize=40)
    im1 = ax[0, 0].imshow(sim[0], interpolation="none", cmap=cm.viridis, vmin=0)
    ax[0, 0].set_title("IPTG")
    divider = make_axes_locatable(ax[0, 0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    f.colorbar(im1, cax=cax, shrink=0.8)

    im2 = ax[0, 1].imshow(sim[1], interpolation="none", cmap=cm.viridis, vmin=0)
    ax[0, 1].set_title("Nutrients")
    divider = make_axes_locatable(ax[0, 1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    f.colorbar(im2, cax=cax, shrink=0.8)

    im3 = ax[1, 0].imshow(sim[2], interpolation="none", cmap=cm.viridis, vmin=0)
    ax[1,0].set_title("Receivers")
    divider = make_axes_locatable(ax[1, 0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    f.colorbar(im3, cax=cax, shrink=0.8)

    im5 = ax[1, 1].imshow(sim[3], interpolation="none", cmap=cm.viridis, vmin=0 )
    ax[1, 1].set_title("GFP")
    divider = make_axes_locatable(ax[1, 1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    f.colorbar(im5, cax=cax, shrink=0.8)

    return(f)

def run_experiment(params, IPTG_conc, plot = False):

    n0 = 25
    plate_width = 35 #mm
    n_rows = n_cols = 35
    w = plate_width / n_rows #dx, dy


    receiver_radius = 2

    U = np.zeros([4, n_rows, n_cols])

    shape = U.shape
    size = U.size

    U[1] = n0 #set nutrients

    #U[1] = 43400000#nM

    #IPTG
    vol = 1e-6 #  litres
    conc = IPTG_conc #mMolar
    amount = vol*conc # milli moles

    #print('initial amoun of IPTG: ', amount)
    agar_thickness = 3.12 #mm


    init_conc = amount/(w**2*agar_thickness) # conc if all IPTG is put into one grid point mmol/mm^3
    #init_conc = amount/(w**2) # conc if all IPTG is put into one grid point mmol/mm^2
    init_conc *= 1e6 # mM

    #print('initial IPTG conc:', init_conc)


    U[0][int(n_rows/2),int(n_cols/2)] = init_conc

    dist = 4.5 #mm
    centre = plate_width/2


    receiver_pos = [[centre - i * dist, centre] for i in range(1, 4)]
    receiver_pos.extend([[centre + i * dist, centre] for i in range(1, 4)])
    receiver_pos.extend([[centre, centre + i * dist] for i in range(1, 4)])
    receiver_pos.extend([[centre, centre - i * dist] for i in range(1, 4)])


    receiver_coordinates = get_node_coordinates(receiver_pos, receiver_radius, n_rows, n_cols, w)

    #t_final = 20*62  # mins threshold
    t_final = 20*67  # mins threshold

    #t_final = 20 * 47  # mins
    dt = .1
    t_points = int(t_final / dt)

    t = np.arange(0, t_final, dt)
    U_init = U.flatten()
    start_time = time.time()

    func = lambda t, U: model_IPTG_BP(t, U, shape, params, w, receiver_coordinates)

    sim_ivp = solve_ivp(func, [0, t_final], U_init, t_eval=t)

    sim_ivp = sim_ivp.y.reshape(4, n_rows, n_cols, t_points)



    if os.path.isdir(os.getcwd() + "/IPTG_inf/output_cross") is False:
        print("ciao")
        os.makedirs(os.getcwd() + "/IPTG_inf/output_cross")


    n_rows, n_cols = sim_ivp[3,:,:,0].shape



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

def measure_flourescence(U):
    # each well split into 8x8 squares and these used to measure flourescence
    # only the following coordinates (in terms of the measurement grid have colonies

    meas_time = 20 # time between measurements in mins
    dt = 0.1 # dt in simulation

    #colony_coords = [(0,3), (1,3), (2,3), (4,3), (5,3), (6,3), (3,1), (3,2), (3,4), (3,5), (3,6)]
    #distances = [13.5, 9, 4.5, 4.5, 9, 13.5, 9, 4.5, 4.5, 9, 13.5]

    colony_coords = [(0, 3), (1, 3), (2, 3)]
    distances = [13.5, 9, 4.5]

    n_rows, n_cols = U[3,:,:,0].shape


    colony_width = math.ceil(n_rows/8)

    simulated_data = {}


    for t in range(0,U.shape[-1], int(meas_time/dt)):


        u = U[:,:,:,t]

        gfp = u[3]


        #print('GFP:', np.max(gfp))

        data_points = []
        # get the simulation points inside each colony corresponding to each colony
        for i, coords in enumerate(colony_coords):
            distance = distances[i]

            row, col = coords

            pixels = gfp[colony_width*row+1:colony_width*(row+1)+1, colony_width*col:colony_width*(col+1)]

            try:
                simulated_data[distance].append(np.mean(pixels))
            except:
                simulated_data[distance] = []
                simulated_data[distance].append(np.mean(pixels))



    return simulated_data

def run_all_experiments(params, plot=False):

    all_data = {}

    for i, conc in enumerate(IPTG_concs):

        sol = run_experiment(params, conc, plot)

        simulated_data = measure_flourescence(sol)

        if plot:

            plt.subplot(2, 4, i+1)

            plt.plot(np.linspace(0, 20, len(simulated_data[4.5])), simulated_data[4.5], 'red', label = '4.5')
            plt.plot(np.linspace(0, 20, len(simulated_data[9])), simulated_data[9], 'green', label = '9')
            plt.plot(np.linspace(0, 20, len(simulated_data[13.5])), simulated_data[13.5], 'blue', label = '13.5')

            plt.plot(np.linspace(0, 20, len(np.array(all_lab_data[conc][4.5]).T)), np.mean(np.array(all_lab_data[conc][4.5]), axis = 0).T, 'r--', label='4.5')
            plt.plot(np.linspace(0, 20, len(np.array(all_lab_data[conc][4.5]).T)), np.mean(np.array(all_lab_data[conc][9]), axis = 0).T, 'g--', label='9')
            plt.plot(np.linspace(0, 20, len(np.array(all_lab_data[conc][4.5]).T)), np.mean(np.array(all_lab_data[conc][13.5]), axis = 0).T, 'b--',  label='13.5')
            plt.ylim(top=0.3)
            plt.ylabel('GFP mean pixel value per well')
            plt.xlabel('Time (h)')
            plt.legend()
            plt.title(str(conc))


        all_data[conc] = simulated_data
    #if plot: plt.show()

    return all_data

def vector_objective(params, *args):
    '''
    objective in vector form for the PSO library
    :param params:
    :param args:
    :return:
    '''


    errors = []
    #print(params)

    for i in range(len(params)):
        all_sim_data = run_all_experiments(params[i,:])
        error = 0

        for IPTG_conc in IPTG_concs:


            for distance in [4.5, 9.0, 13.5]:
                lab_data = np.array(all_lab_data[IPTG_conc][distance])
                sim_data = np.array(all_sim_data[IPTG_conc][distance])

                diff = lab_data-sim_data

                #error += np.sum(((diff)/(lab_data+0.00001))**2)/len(sim_data)
                error += np.sum(((diff))**2)/len(sim_data)


        errors.append(error)

    return errors




if __name__ == '__main__':

    plot = False
    particle_swarm = True
    threshold = False

    if threshold:

        filepath_TH = os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'),
                                   '201201_IPTGsendersZG_img_data_summary.csv')

        n_points = 62

        high_concs_data = load_spot_data(filepath_TH, n_points)

        filepath_small_concs = os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'),
                                   '210114_IPTGspot_norm_data_summary.csv')

        n_points = 73

        low_concs_data = load_spot_data_small_concs(filepath_small_concs, n_points)
        all_lab_data = {}
        print(low_concs_data.keys())

        for IPTG_conc in [1., 2.5]:
            for distance in [4.5, 9.0, 13.5]:
                # cut off lab data at 5 hours and remove offset
                lab_data = np.array(low_concs_data[IPTG_conc][distance])[:,:62]

                try:
                    all_lab_data[IPTG_conc][distance] = lab_data
                except:
                    all_lab_data[IPTG_conc] = {}
                    all_lab_data[IPTG_conc][distance] = lab_data

        for IPTG_conc in [0., 5.]:
            for distance in [4.5, 9.0, 13.5]:
                # cut off lab data at 5 hours and remove offset
                lab_data = np.array(high_concs_data[IPTG_conc][distance])
                print(lab_data.shape)
                lab_data -= lab_data[:,0].reshape(-1,1)
                lab_data = lab_data

                try:
                    all_lab_data[IPTG_conc][distance] = lab_data
                except:
                    all_lab_data[IPTG_conc] = {}
                    all_lab_data[IPTG_conc][distance] = lab_data

        # IPTG_concs = [0., 5., 10., 50., 100., 500.] #mM
        # IPTG_concs = [0., 1., 2., 3., 4., 5.] #mM
        # IPTG_concs = [0., 0.1, 0.2, 0.3, 0.4, 0.5] #mM
        # IPTG_concs = [0., 0.01, 0.02, 0.03, 0.04, 0.05] #mM
        IPTG_concs = [0., 1., 2.5, 5.]


        bounds = ([1, -1, -4, -3, 2], [6, 2, -1, 2, 4])
        #K_a, D, n0, and max on log

        bounds = ([1, -1, -4, -3], [6, 2, -1, 2]) #K_a, D, n0, rc, and max on log

        if particle_swarm:
            print('script started')

            best_params = None
            best_error = 99999999

            #params = lambda_a, K_a, D
            #params = lambda_a, K_a, D, max
            params = lambda_a, K_a, D, max, n0
            params = lambda_a, K_a, D, max, rc, init_cells, n0
            #params = lambda_a, K_a, D, K_n, lambda_n, Dc, rc, D_n, rho_n, n0
            #params = lambda_a, K_a, D, K_n, lambda_n, Dc, rc, D_n, rho_n, max



            #run_all_experiments((3.17165732, 0.02136636, 0.0780891))
            #run_all_experiments((-0.98923442, -1.17488627, -1.75977476, -1.9750064))
            #run_all_experiments(params)
            #sys.exit()
            t = time.time()

            print(time.time() -t)


            #bounds = np.array([[0, 10],[0, 100], [0, 0.1]])
            #bounds = np.array([[0, 100],[0, 100], [0, 1], [0, 100.]])
            #bounds = np.array([[-2, 2],[-2, 2], [-4, 0], [-2, 2]]) # these on the log scale
            #bounds = np.array([[0.01, 5],[1e-9, 10e-5], [1e-9, 0.1], [1e-9, 1000], [0.01, 5], [1e-6, 1e-3], [1e-7, 1e-3], [0, 0.1], [0.001, 3], [0.0009,20]])
            #bounds = np.array([[0, 100],[0, 1000], [0, 0.01], [0, 100000000], [0, 100], [0, 0.01], [0, 1e-2], [0, 1], [0, 10000000], [0.0009,0.1]])

            #results = optimize.shgo(objective, bounds)
            #results = optimize.minimize(objective, (2.3, 40, 0.03), callback = callback)

            '''
            n_groups = 5
            n_particles = 10
            n_steps = 500
            cs = [2,2]
            swarm = Swarm(None, bounds, n_particles, n_groups, cs, loss_function = objective)
            print('swarm initialised')
            values, positions, ims = swarm.find_minimum(None, None, None, n_steps, None)
            print(values, positions)
            '''

            options = {'c1': 0.5, 'c2': 0.3, 'w':0.9, 'k': 2, 'p': 10}



            # Call instance of PSO
            optimizer = ps.single.GlobalBestPSO(n_particles=50, dimensions=4, options=options, bounds=bounds)

            # Perform optimization
            cost, pos = optimizer.optimize(vector_objective, iters=1000, verbose=True)

        elif plot:
            print('plt)')
            params = [4.06128738, -0.95477729, -1.27222098, -2.60550301,  3.25206051]
            params = [4.06128738, -0.95477729, -1.27222098, -2.60550301,  3.25206051]
            params = [4.07900763e+00, -9.69510656e-01, -2.30685610e+00, -2.25835283e+00, -3.25936963e+00, 5.19359341e-05,2.71848081e+00]
            params = [3.3011735 , -0.99641983, -1.49507146, -0.32270905]
            params = [ 3.62992409, -0.99913788, -2.67173002, -0.22840798]

            params = [ 2.00434884, -0.00518 ,   -1.42864773,  1.99791128] #lambda_a, K_a, D, max


            run_all_experiments(params, plot = True)

    else:

        #load high concs data
        filepath_BP = os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'),
                                   '201124_IPTGsendersZBD_img_data_summary.csv')

        n_points = 67

        data = load_spot_data(filepath_BP, n_points)

        all_lab_data = {}

        IPTG_concs = [0., 5., 10, 50., 100., 500.]

        for IPTG_conc in IPTG_concs:
            for distance in [4.5, 9.0, 13.5]:

                lab_data = np.array(data[IPTG_conc][distance])

                lab_data -= lab_data[:, 0].reshape(-1, 1)


                try:
                    all_lab_data[IPTG_conc][distance] = lab_data
                except:
                    all_lab_data[IPTG_conc] = {}
                    all_lab_data[IPTG_conc][distance] = lab_data


        #load low concs data
        filepath_small_concs = os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'),
                                            '210114_IPTGspot_norm_data_summary.csv')

        n_points = 73

        low_concs_data = load_spot_data_small_concs(filepath_small_concs, n_points)

        print(low_concs_data.keys())

        for IPTG_conc in [1., 2.5]:
            for distance in [4.5, 9.0, 13.5]:

                lab_data = np.array(low_concs_data[IPTG_conc][distance])[:, :67]
                lab_data -= lab_data[:, 0].reshape(-1, 1)
                try:
                    all_lab_data[IPTG_conc][distance] = lab_data
                except:
                    all_lab_data[IPTG_conc] = {}
                    all_lab_data[IPTG_conc][distance] = lab_data

        IPTG_concs = [0., 1., 2.5, 5.]

        bounds = ([1, -5, 1, -5, -4, -9, -3], [10, 3, 10, 3, -1, 2, 2])   #na, ka, nr, kr, D, min, max
        #bounds = ([1.58646428, -0.6564445, 7.99974394, -2.14317698,-4, -3.88586878, -0.37292162], [1.58646428, -0.6564445, 7.99974394, -2.14317698, -1, -3.88586878, -0.37292162])   #na, ka, nr, kr, D, min, max
        #bounds = ([ -4, -5, -5], [ -1, 2, 2])   # D, min, max
        #bounds = ([-4], [-1]) #D
        if particle_swarm:
            print('script started')

            best_params = None
            best_error = 99999999


            t = time.time()

            print(time.time() - t)

            options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9, 'k': 2, 'p': 10}

            # Call instance of PSO
            optimizer = ps.single.GlobalBestPSO(n_particles=100, dimensions=7, options=options, bounds=bounds)
            #optimizer = ps.single.GlobalBestPSO(n_particles=20, dimensions=3, options=options, bounds=bounds)

            # Perform optimization
            cost, pos = optimizer.optimize(vector_objective, iters=500, verbose=True)


        elif plot:


            params = [ 1.7191045,  -3.15004698,  4.96872748, -2.25785597, -1.76941789, -2.96793219, -1.17898362] #error: 0.030333004614975734

            # na, ka, nr, kr, D, min, max = params
            params = [1.68518111, - 2.38637506,  5.97079153, - 2.42386853, - 1.73989378, - 2.58806744, - 0.52328954] #error: 0.023180550434634687

            params = [-1.98524167] #D error: 0.09150940336653682

            params = [-2.08820259, -3.32390486,  0.43601529] #D, min, max = params



            params = [-1.68410385, -2.38795859,  0.14782767]#error: 0.030189356066458846


            #na, ka, nr, kr, D, min, max = params

            #error: 0.0213708368905001
            params =[2.59680119, - 2.91649705,  5.86890189 ,- 2.24984255, - 1.82864745 ,- 2.15809078,- 1.07663163]

            plt.figure()
            run_all_experiments(params, plot=True)
            plt.show()









