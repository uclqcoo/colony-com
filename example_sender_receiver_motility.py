from plate import Plate
from species import Species
import numpy as np
import helper_functions as hf


def main():
    ## experimental parameters
    w = 0.5

    D_p = 500 * (6 * 10**-5)    # max chemotaxis rate (450 um2/s)
    D_p0 = 10 * (6 * 10**-5)    # min chemotaxis rate (10 um2/s)
    D_h = 400 * (6 * 10**-5)   # AHL diffusion rate
    D_n = 800 * (6 * 10**-5)  # nutrient diffusion coeff (800 um2/s)
    D_b = 100 * (6 * 10**-5)   # bacteriocin diffusion rate

    gamma = 0.7 / 60 # growth rate (0.7 hr-1)
    beta = 1.04 / 60 # AHL half-life
    m = 20  #
    n_0 = 15E8  #
    k_n = 1 #
    K_n = 1E9   #
    K_h = 4E8   #
    alpha = beta    # AHL production rate
    omega = 5 * 10**6
    min_kb = 1E-20
    max_kb = 1E-16

    dim_mm = 40
    dim = int(dim_mm / w)
    # environment_size = (int(dim*(2/3)), dim)
    environment_size = (dim, dim)
    plate = Plate(environment_size)

    ## add nutrient to the plate
    U_n = np.ones(environment_size) * n_0
    n = Species("n", U_n)
    def n_behaviour(species, params):
        ## unpack params
        w, D_p, D_p0, D_h, D_n, D_b, gamma, beta, m, n_0, k_n, K_n, K_h, alpha, omega, min_kb, max_kb = params

        dn = D_n * hf.ficks(species['n'], w) - (k_n * gamma * species['n']**2 * species['p']) / (species['n']**2 + K_n**2) - (k_n * gamma * species['n']**2 * species['s']) / (species['n']**2 + K_n**2)
        return dn
    n.set_behaviour(n_behaviour)
    plate.add_species(n)

    ## add sender strain to the plate
    U_s = np.zeros(environment_size)
    # for i in np.linspace(5, environment_size[0]-5, 5):
    #     for j in np.linspace(5, environment_size[1]-5, 5):
    #         # if (i == 25) & (j == 25):
    #         #     continue
    #         U_s[int(i), int(j)] = 5E7
    # U_s[int(dim / 2), int(dim / 2)] = 1E7
    # U_s[int(dim / 2), int(dim / 4)] = 1E7
    # U_s[int(dim / 4), int(dim / 2)] = 1E7
    # U_s[int(dim / 4), int(dim / 4)] = 1E7
    s = Species("s", U_s)
    def s_behaviour(species, params):
        ## unpack params
        w, D_p, D_p0, D_h, D_n, D_b, gamma, beta, m, n_0, k_n, K_n, K_h, alpha, omega, min_kb, max_kb = params

        ds = hf.ficks(D_p0 * species['s'], w) + (gamma * species['n']**2 * species['s']) / (species['n']**2 + K_n**2)
        return ds
    s.set_behaviour(s_behaviour)
    plate.add_species(s)

    ## add strain to the plate
    # U_p = np.zeros(environment_size)
    # for i in np.linspace(5, environment_size[0]-5, 9):
    #     for j in np.linspace(5, environment_size[1]-5, 9):
    #         # if (i == 25) & (j == 25):
    #         #     continue
    #         U_p[int(i), int(j)] = 1E7

    U_p = np.ones(environment_size) * 2E8
    p = Species("p", U_p)
    def p_behaviour(species, params):
        ## unpack params
        w, D_p, D_p0, D_h, D_n, D_b, gamma, beta, m, n_0, k_n, K_n, K_h, alpha, omega, min_kb, max_kb = params

        mu_h = D_p0 + (D_p - D_p0) * species['h']**m / (species['h']**m + K_h**m)
        dp = hf.ficks(mu_h * species['p'], w) + (gamma * species['n']**2 * species['p']) / (species['n']**2 + K_n**2)
        return dp
    p.set_behaviour(p_behaviour)
    plate.add_species(p)

    ## add AHL to plate
    U_h = np.zeros(environment_size)
    U_h[int(dim / 2), :] = 5E9
    h = Species("h", U_h)
    def h_behaviour(species, params):
        ## unpack params
        w, D_p, D_p0, D_h, D_n, D_b, gamma, beta, m, n_0, k_n, K_n, K_h, alpha, omega, min_kb, max_kb = params

        dh = D_h * hf.ficks(species['h'], w) + alpha * species['s'] - beta * species['h']
        return dh
    h.set_behaviour(h_behaviour)
    plate.add_species(h)

    plate.plot_plate()

    ## run the experiment
    params = (w, D_p, D_p0, D_h, D_n, D_b, gamma, beta, m, n_0, k_n, K_n, K_h, alpha, omega, min_kb, max_kb)
    sim = plate.run(t_final = 26*60,
                    dt = 1,
                    params = params)

    ## plotting
    tp = np.arange(0, 24*60, 2*60)
    plate.plot_simulation(sim, tp)


main()