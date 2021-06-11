
from functionsDiffusion import *
import sys
import math


# fixed global parameters from Doong et. al. 2017
x_s       = 20              # synthase production rate(au/min)
x_a       = 0.1             # ahl production rate nM/min
x_g       = .1
lambda_a  = 2.3             # hill coeff ahl2
K_a       = 40              # M and M constant for ahl2
D         = 0.03            # AHL diffusion coeff (#mm2/min)
D_a       = 0.3


# Results from Clemens' data
K_a = 17.8
lambda_a = 1.75
min =  4230
max = 54096

def hill_AHL(conc, n, kd, min, max):
    # get rid of the very small negatvie values
    conc[conc<0] = 0


    h = (min + (max-min)*(conc**n/(kd**n + conc**n)))/max

    return h

rho_n     = 3
rc        = 6 * 0.0001
Dc        = 0.0001
rho       = 5 * 0.001
lambda_n  = 2.0
K_n       = 80




def model_one_hidden_layer(t, U_flat, shape):
    U_grid = U_flat.reshape(shape)

    # 0 AHL 1 synthase
    # 1 Arabinose
    # 2 Nutrients
    # 3 Input layer cells
    # 4 AHL 1
    # 5 Hidden layer cells
    # 6 GFP
    # 7 AHL 2 synthase
    # 8 AHL 2
    # 9 Output layer cells

    N = hill(U_grid[2], K_n, lambda_n)

    A1syn_ficks = ficks(U_grid[0], w)
    arabinose_ficks = ficks(U_grid[1], w)
    n_ficks = ficks(U_grid[2], w)
    IL_ficks = ficks(U_grid[3], w)
    A1_ficks = ficks(U_grid[4], w)
    HL_ficks = ficks(U_grid[5], w)
    OL_ficks = ficks(U_grid[9], w)
    A2_ficks = ficks(U_grid[8], w)

    IL = Dc * IL_ficks + rc * N * U_grid[3]
    HL = Dc * HL_ficks + rc * N * U_grid[5]
    OL = Dc * OL_ficks + rc * N * U_grid[9]
    A1syn = x_s * N * hill_AHL(U_grid[1], lambda_a, K_a, min, max) * U_grid[3] # cells produce LuxI (AHL synthase) on induction with arabinose
    A1 = D_a * A1_ficks + (x_a * U_grid[0]) - rho * U_grid[4] # c6 produced by LuxI (AHL synthase)
    arabinose = D * arabinose_ficks + prod_rate*U_grid[3] # add input in rate here
    n = D * n_ficks - rho_n * N * (U_grid[3] + U_grid[5] + U_grid[8]) #nutrients

    A2syn = x_s * N * hill_AHL(U_grid[4], lambda_a, K_a, min, max) * U_grid[5]  # cells produce LuxI (AHL synthase) on induction with arabinose
    A2 = D_a * A2_ficks + (x_a * U_grid[7]) - rho * U_grid[8]  # c6 produced by LuxI (AHL synthase)

    gfp = x_g * N * hill_AHL(U_grid[8], lambda_a, K_a, min, max) * U_grid[9]

    return (np.concatenate((A1syn.flatten(),
                            arabinose.flatten(),
                            n.flatten(),
                            IL.flatten(),
                            A1.flatten(),
                            HL.flatten(),
                            gfp.flatten(),
                            A2syn.flatten(),
                            A2.flatten(),
                            OL.flatten())))

colony_radius = 0.75
w = 0.375  # dx, dy #TODO: this should probably be an argument to model small so it can be set in the same place as n_rows, n_cols

prod_rate = 3.8e-3/60  # per colony, converted from n
prod_rate /= math.pi*colony_radius **2 # per unit area

def run_neural():

    # load network
    #positions = np.load("/home/neythen/Desktop/Projects/synbiobrain/neural_network/network_out/fitted_diffusion_test/best_grid.npy", allow_pickle = True)
    #print(positions)

    # If you double n_rows, half w to get the same sized grid

    n_rows = n_cols = 60  # with w = 0.75 gives 20mm x20mm

    # setup 7 concentrations on a grid 30 x 30
    U = np.zeros([10, n_rows, n_cols])

    shape = U.shape
    size = U.size

    U[2] = 100  # set nutrients

    # sender, positions are now in mm by multiplying by 0.75
    IL_pos = [[1.5+5,2.1+5], [4.2+5,0.8+5]]  # can do multiple senders by adding more coords here
    IL_radius = colony_radius
    IL_coordinates = get_node_coordinates(IL_pos, IL_radius, n_rows, n_cols, w)

    # set initial sneder conc
    IL_rows = IL_coordinates[:, 0]
    IL_cols = IL_coordinates[:, 1]
    U[3][IL_rows, IL_cols] = 1




    # set initial arabinose conc
    U[1][IL_rows, IL_cols] = 500

    # recievers

    HL_pos = [[7.1+5,9.3+5]]

    HL_radius = IL_radius

    HL_coordinates = get_node_coordinates(HL_pos, HL_radius, n_rows, n_cols, w)

    HL_rows = HL_coordinates[:, 0]
    HL_cols = HL_coordinates[:, 1]
    U[5][HL_rows, HL_cols] = 1

    OL_pos = [[5.6+5, 5.9+5]]

    OL_radius = IL_radius

    OL_coordinates = get_node_coordinates(OL_pos, OL_radius, n_rows, n_cols, w)

    rows = OL_coordinates[:, 0]
    cols = OL_coordinates[:, 1]
    U[9][rows, cols] = 1

    t_final = 2000  # mins
    dt = .1
    t_points = int(t_final / dt)

    t = np.arange(0, t_final, dt)
    U_init = U.flatten()
    start_time = time.time()

    sim_ivp = solve_ivp(model_one_hidden_layer, [0, t_final], U_init,
                        t_eval=t, args=(shape,))

    sim_ivp = sim_ivp.y.reshape(10, n_rows, n_cols, t_points)

    save_path = "/neural_network"

    if os.path.isdir(os.getcwd() + save_path + "/output_cross") is False:
        print("ciao")
        os.makedirs(os.getcwd() + save_path + "/output_cross")

    with PdfPages(os.getcwd()+ save_path + "/output_cross/simulation-cross-" + time.strftime("%H%M%S") + ".pdf") as pdf:

        for i in np.arange(0, t_final, 120):
            t = int(i / 60)
            f1 = plot_nn(sim_ivp[:, :, :, i], str(t) + " hours")
            pdf.savefig()
            # plt.show()
            plt.close()

    end_time = time.time()

    # 0 AHL 1 synthase
    # 1 Arabinose
    # 2 Nutrients
    # 3 Input layer cells
    # 4 AHL 1
    # 5 Hidden layer cells
    # 6 GFP
    # 7 AHL 2 synthase
    # 8 AHL 2
    # 9 Output layer cells

    print(end_time - start_time)
    print(sim_ivp[4, IL_rows, IL_cols, -1])

    print(hill_AHL(sim_ivp[1, IL_rows, IL_cols, -1], lambda_a, K_a, min, max))



run_neural()