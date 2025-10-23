import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))) #needed to add this becuase couldn't find files directly?
from plate import Plate
from species import Species
import numpy as np
import helper_functions as hf


def main():
    ## experimental parameters
    w = 0.5                     # sets spatial resolution of grid i.e. what size each simulation grid square represents (mm)

    D_B_max = 500 * (6 * 10**-5)    # max diffusion (chemotaxis) rate of Bacteria B (um2/s)
    D_B_min = 10 * (6 * 10**-5)     # min diffusion (chemotaxis) rate of Bacteria B (um2/s)
    D_A = 400 * (6 * 10**-5)        # diffusion rate of AHL A (um2/s)
    D_N = 800 * (6 * 10**-5)        # diffusion rate of nutrient N (um2/s)

    g_B_max = 0.7 / 60               # max growth rate of Bacteria B (hr-1)
    q_A = 1.04 / 60                  # degradation rate of AHL ()
    lam = 20                          # hill coefficient ()
    N_0 = 15E8                       # initial nutrient concentration (umol/L)
    gamma = 1                        # yield coefficient of Bacteria B on nutrient ()
    k_n = 1/gamma                    # inverse yield coefficient of Bacteria B on nutrient ()
    K_N = 1E9                         # nutrient concentration for half-maximal growth ()
    K_A = 4E8                         # AHL concentration for half-maximal repression of motility ()
    p_A = q_A                        # AHL production rate

    dim_mm = 90                        # size of environment (mm)
    dim = int(dim_mm / w)              # size of environment (in grid squares)
    environment_size = (dim, dim)      # make environment square
    plate = Plate(environment_size)

    ## add nutrient to the plate
    U_n = np.ones(environment_size) * N_0 #creates uniform distribution of nutrient, value is N_0 at each grid square
    n = Species("n", U_n) # creates nutrient species
    # define behaviour of nutrient
    def n_behaviour(t, species, params):
        ## unpack params
        w, D_B_max, D_B_min, D_A, D_N, g_B_max, q_A, lam, N_0, k_n, K_N, K_A, p_A = params
        ## define behaviour
        dn = D_N * hf.ficks(species['n'], w) - (k_n * g_B_max * species['n']**2 * species['p']) / (species['n']**2 + K_N**2)
        return dn
    n.set_behaviour(n_behaviour)
    plate.add_species(n)

    ## add strain to the plate
    positions = [3/1, 3/2] # where to place bacteria - at 1/3 and 2/3 height of the plate in the centre
    #positions = [2/1]  # where to place bacteria - center 
    U_p = np.zeros(environment_size) #creates empty grid
    for p in positions:
        for r in np.arange(3./w, -0.001/w, -1):  # makes initial bacteria spot of radius 3mm
            for i in np.arange((dim/p) - r, (dim/p) + r): # places spot at height dim/p
                for j in np.arange((dim / 2) - r, (dim / 2) + r): # places spot in the centre width-wise
                    U_p[int(i), int(j)] = 2 * np.exp(-(r*w) ** 2 / 4) * 10 ** 8 # sets initial bacteria density, Gaussian distribution

    #U_p[50, 50] = 2E8 # alternative way to set initial bacteria density, single point in centre of plate

    p = Species("p", U_p) # creates bacteria species
    # define behaviour of bacteria
    def p_behaviour(t, species, params):
        ## unpack params
        w, D_B_max, D_B_min, D_A, D_N, g_B_max, q_A, lam, N_0, k_n, K_N, K_A, p_A = params
        ## define behaviour
        mu_h = (D_B_max + D_B_min * (species['h'] / K_A)**lam) / (1 + (species['h'] / K_A)**lam)
        #hill_B = hf.leak_hill(species['A'], K_A, lam, D_B_min, D_B_max) # alternative way to define hill function
        dp = hf.ficks(mu_h * species['p'], w) + (g_B_max * species['n']**2 * species['p']) / (species['n']**2 + K_N**2)
        return dp
    p.set_behaviour(p_behaviour)
    plate.add_species(p)

    ## add AHL to plate
    U_h = np.zeros(environment_size) #creates empty grid
    h = Species("h", U_h) # creates AHL species
    # define behaviour of AHL
    def h_behaviour(t, species, params):
        ## unpack params
        w, D_B_max, D_B_min, D_A, D_N, g_B_max, q_A, lam, N_0, k_n, K_N, K_h, p_A = params
        ## define behaviour
        dh = D_A * hf.ficks(species['h'], w) + p_A * species['p'] - q_A * species['h']
        return dh
    h.set_behaviour(h_behaviour)
    plate.add_species(h)

    # plate.plot_plate()    #uncomment to see initial conditions

    ## run the experiment
    params = (w, D_B_max, D_B_min, D_A, D_N, g_B_max, q_A, lam, N_0, k_n, K_N, K_A, p_A)
    #run simulation
    sim = plate.run(t_final=2000,
                    dt=10,
                    params=params)

    ## plotting
    plate.plot_simulation(sim, 10) #plots every 10th time point

    # ## make video of P over time
    # import matplotlib.pyplot as plt
    # import matplotlib.animation as animation
    # from matplotlib import cm
    #
    # plate_view = sim[1]
    #
    # fig, ax = plt.subplots()
    # plt.axis('off')
    # ims = []
    # for idx in range(plate_view.shape[2]):
    #     im = ax.imshow(plate_view[:, :, idx],
    #                    interpolation="none",
    #                    cmap=cm.gist_gray,
    #                    vmin=0,
    #                    vmax=np.max(plate_view),
    #                    animated=True)
    #     ims.append([im])
    #
    # ani = animation.ArtistAnimation(fig, ims, interval=5000/len(ims), blit=True,
    #                                 repeat_delay=1000)
    # ani.save("movie_repressed_taxis.mp4")


main()
