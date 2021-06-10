
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'parameter_estimation'), 'global_optimsiation'))


from functionsDiffusion import *
import numpy as np
import math
import time
from load_spot_data import *
from scipy import optimize
from particle_swarm_optimiser import *
from matplotlib.backends.backend_pdf import PdfPages
from scipy.integrate import odeint
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx


# fixed global parameters from Doong et. al. 2017
Dc        = 0.0001     # diffusion cefficient of cells mm^2/min
rc        = 6e-4 # max growth rate #1/min
#x_g       = .1         # max production of GFP
rho_n     = 3          # yield coefficient 1/min
lambda_n  = 2.0
K_n       = 80 # arbitrary
D_n = 0.03 #diffusion coeff of nutrients mm^2/min

# these two eye balled from data
min = 0
max = 0.27/(5*60)#  units: mena pixel flourescence per minute
print(max)

lambda_a  = 2.3        # hill coeff IPTG
K_a       = 40e-6         # sat constant for IPTG mM
D         = 0.03       # IPTG diffusion coefficitient mm^2/min

n0 = 500
init_cells = 0.5


def hill_IPTG(conc, n, kd, min, max):
    # get rid of the very small negatvie values
    conc[conc<0] = 0
    h = (min + (max-min)*(conc**n/(kd**n + conc**n)))
    return h


def model_IPTG(t, U_flat, shape, params, w):

    #lambda_a, K_a, D = params
    #lambda_a, K_a, D, max = [10**p for p in params] #convert from log scale
    #lambda_a, K_a, D, max,n0 = [10**p for p in params] #convert from log scale
    #lambda_a, K_a, D, max,n0 = params
    lambda_a, K_a, D, max, rc, init_cells, n0 = params

    K_a = 10**K_a
    max = 10**max
    D = 10**D
    rc = 10**rc

    #lambda_a, K_a, D, K_n, lambda_n, Dc, rc, D_n, rho_n, max = params

    U_grid = U_flat.reshape(shape)

    # 0 IPTG
    # 1 Nutrients
    # 2 Receiver
    # 3 GFP

    N = hill(U_grid[1], K_n, lambda_n)
    IPTG_ficks = ficks(U_grid[0], w)
    n_ficks = ficks(U_grid[1], w)
    R_ficks = ficks(U_grid[2], w)

    R = Dc * R_ficks + rc*N*U_grid[2]
    IPTG = D * IPTG_ficks
    n = D_n * n_ficks - rho_n*N*U_grid[2]

    gfp =  N *max* hill(U_grid[0], lambda_a, K_a) * U_grid[2]

    return (np.concatenate((IPTG.flatten(),
                            n.flatten(),
                            R.flatten(),
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

def diverge_event(t, y):

    if np.any([math.isnan(y[i]) for i in range(len(y)) ]) or np.any([y[i] > 1000000 for i in range(len(y)) ]):
        print('event')
        return -0.000001
    else:
        return 0.0000001

diverge_event.terminal = True

def run_experiment(params, IPTG_conc, plot = False):

    n0 = 10**params[-1]
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


    rows = receiver_coordinates[:, 0]
    cols = receiver_coordinates[:, 1]
    U[2][rows, cols] = init_cells

    t_final = 20*62  # mins
    t_final = 20 * 47  # mins
    dt = .1
    t_points = int(t_final / dt)

    t = np.arange(0, t_final, dt)
    U_init = U.flatten()
    start_time = time.time()

    func = lambda t, U: model_IPTG(t, U, shape, params, w)

    sim_ivp = solve_ivp(func, [0, t_final], U_init,
                        t_eval=t)

    sim_ivp = sim_ivp.y.reshape(4, n_rows, n_cols, t_points)



    if os.path.isdir(os.getcwd() + "/IPTG_inf/output_cross") is False:
        print("ciao")
        os.makedirs(os.getcwd() + "/IPTG_inf/output_cross")




    colony_coords = [(0, 3), (2, 3)]
    distances = [13.5,  4.5]

    n_rows, n_cols = sim_ivp[3,:,:,0].shape


    colony_width = math.ceil(n_rows/8)

    simulated_data = {}

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

def run_partial_experiment(params, IPTG_conc):

    plate_width = 35 #mm
    sim_width = plate_width/2
    sim_height = 6

    w = 1
    n_rows = int(sim_height/w)
    n_cols = int(sim_width/w)

    receiver_radius = 2

    U = np.zeros([4, n_rows, n_cols])

    shape = U.shape
    size = U.size

    U[1] = 100  # set nutrients

    #IPTG
    vol = 1 # micro litres
    conc = IPTG_conc #milli Molar
    amount = vol*conc # moles

    agar_thickness = 0.00312 #m

    #for now consider only 2D (ignore the thickness)_
    init_conc = amount/(w**2) # conc if all IPTG is put into one grid point




    dist = 4.5 #mm
    centre = [1,0]

    U[0][centre[0], centre[1]] = init_conc


    receiver_pos = [[centre[0], centre[1] + i * dist] for i in range(1, 4)]

    receiver_coordinates = get_node_coordinates(receiver_pos, receiver_radius, n_rows, n_cols, w)


    rows = receiver_coordinates[:, 0]
    cols = receiver_coordinates[:, 1]
    U[2][rows, cols] = 0.5

    t_final = 20*62  # mins
    t_final = 20*47  # mins

    dt = .1
    t_points = int(t_final / dt)
    print(t_points)

    t = np.arange(0, t_final, dt)
    print(t.shape)
    U_init = U.flatten()
    start_time = time.time()

    sim_ivp = solve_ivp(model_IPTG, [0, t_final], U_init,
                        t_eval=t, args=(shape, params, w))

    sim_ivp = sim_ivp.y.reshape(4, n_rows, n_cols, t_points)


    if os.path.isdir(os.getcwd() + "/IPTG_inf/output_cross") is False:
        print("ciao")
        os.makedirs(os.getcwd() + "/IPTG_inf/output_cross")

    with PdfPages("IPTG_inf/output_cross/simulation-cross-" + time.strftime("%H%M%S") + ".pdf") as pdf:

        for i in np.arange(0, t_final+1, 120):
            t = int(i / 60)
            f1 = plot_IPTG(sim_ivp[:, :, :, i], str(t) + " hours")
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

def measure_partial_flourescence(U):
    # each well split into 8x8 squares and these used to measure flourescence
    # only the following coordinates (in terms of the measurement grid have colonies

    meas_time = 20 # time between measurements in mins
    dt = 0.1 # dt in simulation

    #colony_coords = [(0,3), (1,3), (2,3), (4,3), (5,3), (6,3), (3,1), (3,2), (3,4), (3,5), (3,6)]
    #distances = [13.5, 9, 4.5, 4.5, 9, 13.5, 9, 4.5, 4.5, 9, 13.5]

    colony_coords = [(0, 1), (0, 2), (0, 3)]
    distances = [ 4.5, 9, 13.5]

    n_rows, n_cols = U[3,:,:,0].shape

    colony_width = math.ceil(n_rows/5)

    simulated_data = {}

    for t in range(0,U.shape[-1], int(meas_time/dt)):


        u = U[:,:,:,t]
        gfp = u[3]
        data_points = []
        # get the simulation points inside each colony corresponding to each colony
        for i, coords in enumerate(colony_coords):
            distance = distances[i]

            row, col = coords

            pixels = gfp[colony_width*row:colony_width*(row+1), colony_width*col:colony_width*(col+1)]

            try:
                simulated_data[distance].append(np.mean(pixels))
            except:
                simulated_data[distance] = []
                simulated_data[distance].append(np.mean(pixels))



    return simulated_data

def run_all_experiments(params, plot=False):

    IPTG_concs = [0., 5., 10., 50., 100., 500.] #mM



    all_data = {}
    #plt.figure()
    for i, conc in enumerate(IPTG_concs):
        sol = run_experiment(params, conc, plot)

        simulated_data = measure_flourescence(sol)

        if plot:
            plt.subplot(2, 3, i+1)

            plt.plot(np.linspace(0, 15, len(simulated_data[4.5])), simulated_data[4.5], 'orange', label = '4.5')
            plt.plot(np.linspace(0, 15, len(simulated_data[9])), simulated_data[9], 'green', label = '9')
            plt.plot(np.linspace(0, 15, len(simulated_data[13.5])), simulated_data[13.5], 'blue', label = '13.5')

            plt.plot(np.linspace(0, 15, len(np.array(all_lab_data[conc][4.5]).T)), np.array(all_lab_data[conc][4.5]).T, 'r--', label='4.5')
            plt.plot(np.linspace(0, 15, len(np.array(all_lab_data[conc][4.5]).T)), np.array(all_lab_data[conc][9]).T, 'g--', label='9')
            plt.plot(np.linspace(0, 15, len(np.array(all_lab_data[conc][4.5]).T)), np.array(all_lab_data[conc][13.5]).T, 'b--',  label='13.5')
            plt.ylim(top=0.7)
            plt.ylabel('GFP mean pixel value per well')
            plt.xlabel('Time (h)')
            plt.legend()
            plt.title(str(conc))



        all_data[conc] = simulated_data
    if plot: plt.show()

    return all_data


def objective(params, *args):

    error = 0
    print()
    print(params)
    all_sim_data = run_all_experiments(params)

    for IPTG_conc in [0., 5., 10., 50., 100., 500.]:
        for distance in [4.5, 9.0, 13.5]:
            lab_data = np.array(all_lab_data[IPTG_conc][distance])
            sim_data = np.array(all_sim_data[IPTG_conc][distance])

            diff = lab_data-sim_data

            error += np.sum((lab_data-sim_data)**2)/len(sim_data)
    print('error:', error)

    return error


def vector_objective(params, *args):
    '''
    objective in vector form for the PSO library
    :param params:
    :param args:
    :return:
    '''


    errors = []
    print(params)

    for i in range(len(params)):
        all_sim_data = run_all_experiments(params[i,:])
        error = 0
        for IPTG_conc in [0., 5., 10., 50., 100., 500]:
            for distance in [4.5, 9.0, 13.5]:
                lab_data = np.array(all_lab_data[IPTG_conc][distance])
                sim_data = np.array(all_sim_data[IPTG_conc][distance])

                diff = lab_data-sim_data

                error += np.sum((lab_data-sim_data)**2)/len(sim_data)

        with open('particle_swarm1.csv', 'a') as file:
            print('error:', error, 'params:', params[i,:])
            file.write(str(error) + ',' + ','.join([str(p) for p in params[i,:]]) + '\n')

        errors.append(error)

    return errors



def callback(xk, state):
    print(xk)
    print(state.fun)


if __name__ == '__main__':

    plot = True

    particle_swarm = False
    brute_force = False


    filepath_TH = os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'),
                               '201201_IPTGsendersZG_img_data_summary.csv')

    n_points = 62

    all_lab_data = load_spot_data(filepath_TH, n_points)

    for IPTG_conc in [0., 5., 10., 50., 100., 500.]:
        for distance in [4.5, 9.0, 13.5]:
            # cut off lab data at 5 hours and remove offset
            lab_data = np.array(all_lab_data[IPTG_conc][distance])

            lab_data -= lab_data[0,0]
            lab_data = lab_data[:, 15:]
            all_lab_data[IPTG_conc][distance] = lab_data


    bounds = ([1, -1, -4, -3, 2], [6, 2, -1, 2, 4])
    #K_a, D, n0, and max on log

    bounds = ([1, -1, -4, -3, -5, 0, 2], [6, 2, -1, 2, -1, 0.5, 4]) #K_a, D, n0, rc, and max on log


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
        optimizer = ps.single.GlobalBestPSO(n_particles=50, dimensions=7, options=options, bounds=bounds)

        # Perform optimization
        cost, pos = optimizer.optimize(vector_objective, iters=1000, verbose=True)
    elif brute_force:


        with open("brute_force1.csv", "a") as file:

            for i in range(10000):
                params = np.random.uniform(bounds[0], bounds[1])

                error = objective(params)

                file.write(str(error) + ',' + ','.join([str(p) for p in params])+'\n')

    elif plot:
        params = [4.06128738, -0.95477729, -1.27222098, -2.60550301,  3.25206051]
        params = [4.06128738, -0.95477729, -1.27222098, -2.60550301,  3.25206051]
        params = [4.07900763e+00, -9.69510656e-01, -2.30685610e+00, -2.25835283e+00, -3.25936963e+00, 5.19359341e-05,2.71848081e+00]
        #params = [ 3.15749657, -0.72598656, -1.63022651, -2.56271167, -2.55335996 , 0.28575791,2.96582322]


        run_all_experiments(params, plot = True)









