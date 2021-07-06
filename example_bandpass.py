from plate import Plate
from species import Species
import numpy as np
import math
import helper_functions as hf


def main():
    ## experimental parameters
    D_N = 1e-4
    mu_max = 0.03
    K_mu = 1
    gamma = 1E12
    D_A = 5e-6

    ## From Zong paper
    alpha_T = 6223
    beta_T = 12.8
    K_IT = 1400  # 1.4e-6 M @ 1 molecule per nM
    n_IT = 2.3
    K_lacT = 15719
    T7_0 = 0
    alpha_R = 8025
    beta_R = 30.6
    K_IR = 1200  # 1.2e-6 M @ 1 molecule per nM
    n_IR = 2.2
    K_lacR = 14088
    R_0 = 0
    alpha_G = 16462
    beta_G = 19
    n_A = 1.34
    K_A = 2532
    n_R = 3.9
    K_R = 987

    environment_size = (32, 48)
    w = 1

    # environment_size = (16, 24)
    # w = 2

    # environment_size = (8, 12)
    # w = 4

    plate = Plate(environment_size)

    ## add nutrient to the plate
    U_N = np.ones(environment_size) * 4
    N = Species("N", U_N)
    def N_behaviour(t, species, params):
        ## unpack params
        w, D_N, mu_max, K_mu, gamma, D_A, \
        alpha_T, beta_T, K_IT, n_IT, K_lacT, T7_0, \
        alpha_R, beta_R, K_IR, n_IR, K_lacR, R_0, \
        alpha_G, beta_G, n_A, K_A, n_R, K_R = params

        n = D_N * hf.ficks(species['N'], w) - mu_max * K_mu / (K_mu + species['N']) * species['X'] / gamma
        return n
    N.set_behaviour(N_behaviour)
    plate.add_species(N)

    ## add one strain to the plate
    U_X = np.ones(environment_size) * 1e8
    strain = Species("X", U_X)
    def X_behaviour(t, species, params):
        ## unpack params
        w, D_N, mu_max, K_mu, gamma, D_A, \
        alpha_T, beta_T, K_IT, n_IT, K_lacT, T7_0, \
        alpha_R, beta_R, K_IR, n_IR, K_lacR, R_0, \
        alpha_G, beta_G, n_A, K_A, n_R, K_R = params

        x = mu_max * K_mu / (K_mu + species['N']) * species['X']
        return x
    strain.set_behaviour(X_behaviour)
    plate.add_species(strain)

    ## add IPTG to plate
    U_A = np.ones(environment_size)
    # U_A[int(environment_size[0]/2), int(environment_size[1]/2)] = 6e15
    U_A[int(environment_size[0]/2), int(environment_size[1]/2 - 4 / w)] = 6e15
    U_A[int(environment_size[0]/2), int(environment_size[1]/2 + 4 / w)] = 6e15
    A = Species("A", U_A)
    def A_behaviour(t, species, params):
        ## unpack params
        w, D_N, mu_max, K_mu, gamma, D_A, \
        alpha_T, beta_T, K_IT, n_IT, K_lacT, T7_0, \
        alpha_R, beta_R, K_IR, n_IR, K_lacR, R_0, \
        alpha_G, beta_G, n_A, K_A, n_R, K_R = params

        a = D_A * hf.ficks(species['A'], w)
        return a
    A.set_behaviour(A_behaviour)
    plate.add_species(A)

    ## add GFP to plate
    U_G = np.ones(environment_size)
    G = Species("G", U_G)
    def G_behaviour(t, species, params):
        ## unpack params
        w, D_N, mu_max, K_mu, gamma, D_A, \
        alpha_T, beta_T, K_IT, n_IT, K_lacT, T7_0, \
        alpha_R, beta_R, K_IR, n_IR, K_lacR, R_0,\
        alpha_G, beta_G, n_A, K_A, n_R, K_R = params

        mu = mu_max * K_mu / (K_mu + species['N'])

        T7 = alpha_T + beta_T - alpha_T * K_lacT / (1 + (species['A'] / K_IT)**n_IT + K_lacT) + T7_0 * np.exp(-mu * t)
        R = alpha_R + beta_R - alpha_R * K_lacR / (1 + (species['A'] / K_IR)**n_IR + K_lacR) + R_0 * np.exp(-mu * t)
        # R = 0

        dGFP = (alpha_G * mu * T7**n_A / (K_A**n_A + T7**n_A) * K_R**n_R / (K_R**n_R + R**n_R) + beta_G * mu - species['G'] * mu)

        return dGFP
    G.set_behaviour(G_behaviour)
    plate.add_species(G)

    ## run the experiment
    params = (w, D_N, mu_max, K_mu, gamma, D_A,
              alpha_T, beta_T, K_IT, n_IT, K_lacT, T7_0,
              alpha_R, beta_R, K_IR, n_IR, K_lacR, R_0,
              alpha_G, beta_G, n_A, K_A, n_R, K_R)
    sim = plate.run(t_final = 24*60 + 1,
                    dt = 1,
                    params = params)

    ## plotting
    plate.plot_simulation(sim, 3, 'log10')

main()