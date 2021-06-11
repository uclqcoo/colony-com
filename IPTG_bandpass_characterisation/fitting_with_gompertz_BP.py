
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'IPTG_characterisation'))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'IPTG_characterisation'))
sys.path.append(os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'parameter_estimation'), 'global_optimsiation'))

from fitting_with_gompertz import *
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
from growth_fitting import gompertz


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



if __name__ == '__main__':

    plot = True

    particle_swarm = False
    brute_force = False


    filepath_BP = os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'),
                               '201124_IPTGsendersZBD_img_data_summary.csv')

    n_points = 67

    all_lab_data = load_spot_data(filepath_BP, n_points)






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
    elif brute_force:


        with open("brute_force1.csv", "a") as file:

            for i in range(10000):
                params = np.random.uniform(bounds[0], bounds[1])

                error = objective(params)

                file.write(str(error) + ',' + ','.join([str(p) for p in params])+'\n')

    elif plot:
        print('plt)')
        params = [4.06128738, -0.95477729, -1.27222098, -2.60550301,  3.25206051]
        params = [4.06128738, -0.95477729, -1.27222098, -2.60550301,  3.25206051]
        params = [4.07900763e+00, -9.69510656e-01, -2.30685610e+00, -2.25835283e+00, -3.25936963e+00, 5.19359341e-05,2.71848081e+00]
        params = [3.3011735 , -0.99641983, -1.49507146, -0.32270905]
        params = [ 3.62992409, -0.99913788, -2.67173002, -0.22840798]

        params = [ 2.00434884, -0.00518 ,   -1.42864773,  1.99791128]


        run_all_experiments(params, plot = True)









