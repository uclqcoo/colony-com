from functionsDiffusion import *


def main():
    def model_small(t, U_flat, shape):
        x_s = 20  # synthase production rate(au/min)
        x_a = 0.1
        x_g = .1
        lambda_a = 2.3  # hill coeff ahl2
        K_a = 40  # M and M constant for ahl2
        D = 0.03  # AHL diffusion coeff (#mm2/min)
        D_a = .3
        w = 0.75
        rho_n = 3
        rc = 6 * 0.0001
        Dc = 0.0001
        rho = 5 * 0.001

        U_grid = U_flat.reshape(shape)
        N = hill(U_grid[2], 80, 2.0)
        
        LuxI_ficks = ficks(U_grid[0], w)
        arabinose_ficks = ficks(U_grid[1], w)
        n_ficks = ficks(U_grid[2], w)
        S_ficks = ficks(U_grid[3], w)
        c6_ficks = ficks(U_grid[4], w)
        R_ficks = ficks(U_grid[5], w)
        gfp = ficks(U_grid[6], w)

        S = Dc * S_ficks + rc * N * U_grid[3]
        R = Dc * R_ficks + rc * N * U_grid[5]
        LuxI = x_s * N * hill(U_grid[1], lambda_a, K_a) * U_grid[3]
        c6 = D_a * c6_ficks + (x_a * U_grid[0]) - rho * U_grid[4]
        arabinose = D * arabinose_ficks
        n = D * n_ficks - rho_n * N * (U_grid[3] + U_grid[5])
        gfp = x_g * N * hill(U_grid[4], lambda_a, K_a) * U_grid[5]

        return(np.concatenate((LuxI.flatten(), arabinose.flatten(), n.flatten(),
                               S.flatten(), c6.flatten(), R.flatten(),
                               gfp.flatten())))

    grid_size = 900
    U = np.zeros(grid_size * 7)
    U = U.reshape(7, 30, 30)

    shape = U.shape
    size = U.size

    U[2] = 100

    U[3][14, 14] = 0.5
    U[3][14, 15] = 0.5
    U[3][15, 14] = 0.5
    U[3][15, 15] = 0.5

    U[1][14, 14] = 5
    U[1][14, 15] = 5
    U[1][15, 14] = 5
    U[1][15, 15] = 5

    i = 0
    for j in np.arange(1, 4):
        U[5][11 - i, 14] = 0.5
        U[5][11 - i, 15] = 0.5
        U[5][12 - i, 14] = 0.5
        U[5][12 - i, 15] = 0.5
        i = i + 3
    i = 0
    for j in np.arange(1, 4):
        U[5][17 + i, 14] = 0.5
        U[5][17 + i, 15] = 0.5
        U[5][18 + i, 14] = 0.5
        U[5][18 + i, 15] = 0.5
        i = i + 3

    i = 0
    for j in np.arange(1, 4):
        U[5][14, 11 - i] = 0.5
        U[5][14, 12 - i] = 0.5
        U[5][15, 11 - i] = 0.5
        U[5][15, 12 - i] = 0.5
        i = i + 3

    i = 0
    for j in np.arange(1, 4):
        U[5][14, 17 + i] = 0.5
        U[5][14, 18 + i] = 0.5
        U[5][15, 17 + i] = 0.5
        U[5][15, 18 + i] = 0.5
        i = i + 3

    t_final = 1000  # mins
    dt = .1
    t_points = int(t_final / dt)

    t = np.arange(0, t_final, dt)
    U_init = U.flatten()
    start_time = time.time()

    sim_ivp = solve_ivp(model_small, [0, t_final], U_init,
                        t_eval=t, args=(shape,))

    sim_ivp = sim_ivp.y.reshape(7, 30, 30, t_points)

    if os.path.isdir(os.getcwd() + "/simulation_pdfs") is False:
        print("ciao")
        os.mkdir(os.getcwd() + "/simulation_pdfs")

    with PdfPages("simulation_pdfs/simulation_" + time.strftime("%H%M%S") + ".pdf") as pdf:

        for i in np.arange(0, 901, 120):
            t = int(i / 60)
            f1 = multi_plots(sim_ivp[:, :, :, i], str(t) + " hours")
            pdf.savefig()
            # plt.show()
            plt.close()

    end_time = time.time()
    print(end_time - start_time)


main()
