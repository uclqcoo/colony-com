from functionsDiffusion import *

# fixed global parameters from Doong et. al. 2017
x_s = 20  # synthase production rate(au/min)
x_a = 0.1
x_g = .1
lambda_a = 2.3  # hill coeff ahl2
K_a = 40  # M and M constant for ahl2
D = 0.03  # AHL diffusion coeff (#mm2/min)
D_a = 0.3

rho_n = 3
rc = 6 * 0.0001
Dc = 0.0001
rho = 5 * 0.001
lambda_n = 2.0
K_n = 80


def model_small(t, U_flat, shape):
    U_grid = U_flat.reshape(shape)

    # 0 LuxI
    # 1 Arabinose
    # 2 Nutrients
    # 3 Sender
    # 4 C6
    # 5 Receiver
    # 6 GFP

    N = hill(U_grid[2], K_n, lambda_n)

    LuxI_ficks = ficks(U_grid[0], w)
    arabinose_ficks = ficks(U_grid[1], w)
    n_ficks = ficks(U_grid[2], w)
    S_ficks = ficks(U_grid[3], w)
    c6_ficks = ficks(U_grid[4], w)
    R_ficks = ficks(U_grid[5], w)

    S = Dc * S_ficks + rc * N * U_grid[3]
    R = Dc * R_ficks + rc * N * U_grid[5]
    LuxI = x_s * N * hill(U_grid[1], lambda_a, K_a) * U_grid[3]
    c6 = D_a * c6_ficks + (x_a * U_grid[0]) - rho * U_grid[4]
    arabinose = D * arabinose_ficks
    n = D * n_ficks - rho_n * N * (U_grid[3] + U_grid[5])
    gfp = x_g * N * hill(U_grid[4], lambda_a, K_a) * U_grid[5]

    return (np.concatenate((LuxI.flatten(),
                            arabinose.flatten(),
                            n.flatten(),
                            S.flatten(),
                            c6.flatten(),
                            R.flatten(),
                            gfp.flatten())))


w = 0.375  # dx, dy #TODO: this should probably be an argument to model small so it can be set in the same place as n_rows, n_cols
def run_characterisation_exp():
    n_rows = n_cols = 60  # with w = 0.75 gives 20mm x20mm

    all_vertex_numbers = np.arange(n_rows * n_cols).reshape(-1, 1)  # reshpae to colum vector

    all_vertex_positions = get_vertex_positions(all_vertex_numbers, n_rows, n_cols, w)

    # setup 7 concentrations on a grid 30 x 30
    U = np.zeros([7, n_rows, n_cols])

    shape = U.shape
    size = U.size

    U[2] = 100  # set nutrients

    # sender, positions are now in mm by multiplying by 0.75
    sender_pos = [[5, n_cols*w/2]]
    sender_radius = 1.5
    sender_coordinates = get_node_coordinates(all_vertex_positions, sender_pos, sender_radius, n_rows, n_cols)

    # uncomment these lines to see the vertex assignment
    '''
    sender_numbers, sender_indicators = assign_vertices(all_vertex_positions, sender_pos, sender_radius)
    print(sender_indicators.reshape(n_rows, n_cols))
    '''
    # set initial sneder conc

    rows = sender_coordinates[:, 0]
    cols = sender_coordinates[:, 1]
    U[3][rows, cols] = 0.5

    # set initial arabinose conc
    U[1][rows, cols] = 5


    distances = [4.5, 6.39, 9, 10.08, 13.5]

    receiver_pos = [[sender_pos[0][0] + 4.5, n_cols*w/2], [sender_pos[0][0]  + 9, n_cols*w/2], [sender_pos[0][0]  + 13.5, n_cols*w/2]] # jsut do straight ones for now
    receiver_radius = sender_radius

    receiver_coordinates = get_node_coordinates(all_vertex_positions, receiver_pos, receiver_radius, n_rows, n_cols)
    # uncomment these lines to see the vertex assignment
    '''
    receiver_numbers, receiver_indicators = assign_vertices(all_vertex_positions, receiver_pos, receiver_radius)
    print(receiver_indicators.reshape(n_rows, n_cols))
    '''

    rows = receiver_coordinates[:, 0]
    cols = receiver_coordinates[:, 1]
    U[5][rows, cols] = 0.5


    t_final = 21 #hours
    t_final = t_final*60 # mins
    dt = .1
    t_points = int(t_final / dt)

    t = np.arange(0, t_final, dt)
    U_init = U.flatten()
    start_time = time.time()

    sim_ivp = solve_ivp(model_small, [0, t_final], U_init,
                        t_eval=t, args=(shape,))

    sim_ivp = sim_ivp.y.reshape(7, n_rows, n_cols, t_points)

    save_dir = '/characterisation'

    if os.path.isdir(os.getcwd() + save_dir) is False:
        print("ciao")
        os.makedirs(os.getcwd() + save_dir)

    with PdfPages(os.getcwd() + save_dir + "/simulation-cross-" + time.strftime("%H%M%S") + ".pdf") as pdf:

        for i in np.arange(0, t_final, 120):
            t = int(i / 60)
            f1 = multi_plots(sim_ivp[:, :, :, i], str(t) + " hours")
            pdf.savefig()
            # plt.show()
            plt.close()

    end_time = time.time()
    print(end_time - start_time)

run_characterisation_exp()