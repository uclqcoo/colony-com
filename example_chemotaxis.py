from plate import Plate
from species import Species
import numpy as np
import helper_functions as hf


def main():
    ## experimental parameters
    # D = 3E-3  # nutrient diffusion coeff (#mm2/min)
    # rho_n = 0.3  # consumption rate of nutrients by X
    # rc = 1E-2  # growth rate of X on N (N being?)
    # Dc = 1E-5  # cell diffusion coefficient
    # w = 1     # scales size of grid  - when w = 1 environment is 20x20mm
    # Da = 0.03
    # rho_A = 0.1  # production rate of AHL

    D = 3E-3  # nutrient diffusion coeff (#mm2/min)
    rho_n = 0.3  # consumption rate of nutrients by X
    rc = 1E-2  # growth rate of X on N
    Dc = 1E-6  # cell diffusion coefficient
    w = 1       # scales size of grid  - when w = 1 environment is 20x20mm
    Da = 0.03  # diffusion coeff ahl?
    rho_A = 0.1  # production rate of AHL
    mumaxc = 1E-2 # maximum growth of x
    Kc =  0.5      # nutrient conc for half max growth of x


    environment_size = (20, 20)
    plate = Plate(environment_size)

    ## add nutrient to the plate
    U_N = np.ones(environment_size)
    N = Species("N", U_N)

    def N_behaviour(t, species, params):
        ## unpack params
        D, rho_n, Dc, rc, w, rho_A, Da = params
        n = D * hf.ficks(species['N'], w) - rho_n * species['N'] * (species['R'])
        return n

    N.set_behaviour(N_behaviour)
    plate.add_species(N)

    ## add motile strain to the plate
    U_R = np.zeros(environment_size)
    for i in np.linspace(1, 9, 8):       # where to put cells
        U_R[int(i), int(i)] = 0.001     # how much at each point
    R = Species("R", U_R)

    def R_behaviour(t, species, params):
        ## unpack params
        D, rho_n, Dc, rc, w, rho_A, Da = params
        #r = Dc * hf.leaky_hill(s=species['A'], K=1, lam=2, max=1e2, min=1) * hf.ficks(species['R'], w) + rc * species[
            #'N'] * species['R']    # REPLACE w monod growth

        r = Dc * hf.leaky_hill(s=species['A'], K=1, lam=2, max=1e2, min=1) * hf.ficks(species['R'], w) + hf.monod(s1=species['A'], s2=species['N'], mumax = mumaxc, Kv = Kc )   # REPLACE w monod growth
        return r

    R.set_behaviour(R_behaviour)
    plate.add_species(R)

    ## add AHL to plate
    U_A = np.zeros(environment_size)
    grad_values = np.logspace(-4, 5, environment_size[0])
    for idx, value in enumerate(grad_values):
        U_A[idx, :] = value

    A = Species("A", U_A)

    def A_behaviour(t, species, params):
        ## unpack params
        D, rho_n, Dc, rc, w, rho_A, Da = params
        # a = Da * hf.ficks(species['A'], w) -------- why is this commented out?
        a = 0
        return a

    A.set_behaviour(A_behaviour)
    plate.add_species(A)

    plate.plot_plate()
    print("plotted first figure")

    ## run the experiment
    params = (D, rho_n, Dc, rc, w, rho_A, Da)
    sim = plate.run(t_final=50 * 60,
                    dt=0.1,
                    params=params)
    print("run simulation")

    ## plotting
    plate.plot_simulation(sim, 10)

    print("final plot")


main()
