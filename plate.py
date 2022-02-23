import matplotlib.pyplot as plt  # state-based interface to matplotlib
from matplotlib import cm  # colour maps
from matplotlib import gridspec  # grid layout to place subplots within a figure
from mpl_toolkits.axes_grid1 import make_axes_locatable  # used to create a divider for an axes
import numpy as np  # maths stuff
from scipy.integrate import solve_ivp  # from scipy - integration and ODEs - solve_ivp(fun, t_span, y0[, method,
# t_eval, …]) Solve an initial value problem for a system of ODEs.


class Plate:
    def __init__(self, size):  # __init__ means this is defined for each instance of class not all members of class
        self.size = size  # dimensions of plate (row, column)
        self.species = []  # list of species objects on plate

    def get_size(self):
        return self.size  # return dimensions of plate (row, column)

    def get_num_species(self):
        return len(self.species)  # returns number of species by counting length of list of species

    def get_all_species(self):
        return self.species  # returns list of species on the plate

    def get_species_by_name(self, name):
        for s in self.species:
            if s.get_name() == name:  # get_name  function from species.py
                return s
        else:
            return None  # searches list of species for named species s, if not fount returns none

    def get_all_species_U(self):
        U = np.zeros((self.get_num_species(), self.size[0], self.size[1]))
        # creates 3 dimensional matrix of 0s with dimensions [num_species, row_length, column_length]

        for idx, s in enumerate(self.species):  # idx is index or counter showing current step of iteration, used with enumerate
            U[idx] = s.get_U()
        return U


    def add_species(self, new_species):
        self.species.append(new_species) # adds new species to the end of the list of species on the plate

    def set_species(self, species):
        self.species = species   # used to change list of species by using this function - from externally - this is a better way

    def model(self, t, y, params):
        # reshape vector 'y' into a 3-dimensional matrix with dimensions [num_species, row_length, column_length]
        U = y.reshape(self.get_all_species_U().shape)

        dU = np.zeros(U.shape)

        # for each species, get their U and behaviour function
        species_dict = {}  # {} creates an empty dictionary
        behaviour_dict = {}
        for idx, s in enumerate(self.species):
            species_dict[s.get_name()] = U[idx]
            behaviour_dict[s.get_name()] = s.behaviour

        # for each species, run the behaviour function to determine change of time step
        for idx, s in enumerate(self.species):
            dU[idx] = behaviour_dict[s.get_name()](t, species_dict, params)

        # flatten and return dU
        return dU.flatten()  # .flatten returns a copy of the array flattened in row-major (C-style) order, ie along
        # one whole row then along the whole row below it left to right

    def run(self, t_final, dt, params):
        # get time points
        t = np.arange(0, t_final, dt)  # Return values evenly spaced (start, stop, step)

        # flatten all species U matrix (solver takes state as a vector not matrix)
        U_init = self.get_all_species_U().flatten()

        # numerically solve model
        sim_ivp = solve_ivp(self.model, [0, t_final], U_init,
                            t_eval=t, args=(params,))

        # reshape species into matrix [num_species, row_length, column_length, num_timepoints]
        sim_ivp = sim_ivp.y.reshape(self.get_num_species(),
                                    self.size[0], self.size[1],
                                    len(t))

        return sim_ivp  # ivp = initial value problem

    def plot_simulation(self, sim, num_timepoints, scale='linear', cols=3): # 3 columns of plots

        # returns evenly spaced numbers over a specified interval
        tps = np.linspace(0, sim.shape[3] - 1, num_timepoints) # set time points to plot (not all of the ones calcualted)

        for tp in tps: # time point in time points
            tp = int(tp)  # int() converts to integer

            rows = int(np.ceil(len(self.species) / cols))
            # np.ceil - ceiling function gives first integer greater than or equal to input,
            # cols is location where the output is stored

            # find how many rows needed based on fixed 3 columns

            gs = gridspec.GridSpec(rows, cols)  # grid layout to place subplots within a figure
            fig = plt.figure()  # create a new figure - here empty

            for idx in range(len(self.species)): # for each index in list of numbers the length of the number of species
                ax = fig.add_subplot(gs[idx]) # new plot per species
                if scale == "log10":
                    im = ax.imshow(np.log10(sim[idx, :, :, tp]), interpolation="none",
                                   # .imshow displays data as an image (.im show (array, colour map, vmin and v max
                                   # define data range image covers)
                                   cmap=cm.viridis, vmin=np.min(np.log10(sim[idx, :, :, :])),
                                   vmax=np.max(np.log10(sim[idx, :, :, :])))
                elif scale == "linear":
                    im = ax.imshow(sim[idx, :, :, tp], interpolation="none",
                                   cmap=cm.viridis, vmin=0,
                                   vmax=np.max(sim[idx, :, :, :]))

                ax.set_title(self.species[idx].get_name() + ' : ' + str(tp))  # set title species name : time point

                divider = make_axes_locatable(ax) # Axes divider to calculate location of axes and create a divider
                # for them using existing axes instances.
                cax = divider.append_axes("right", size="5%", pad=0.05)  # adding another axis the same height as the
                # image for the colour bar, with a slight gap in between
                # cax - axes to which the colour bar will be drawn
                fig.colorbar(im, cax=cax, shrink=0.8)

            plt.subplots_adjust(wspace=0.6)  # adjusts subplot layout parameters, wspace - The width of the padding
            # between subplots, as a fraction of the average Axes width.
            fig.savefig('fig_' + str(tp) + '.png')

            fig.show()

    def plot_plate(self, cols=3):
        print("plotting plate")

        rows = int(np.ceil(len(self.species) / cols))  # The ceil of scalar x is the smallest integer i, such that i >= x
        gs = gridspec.GridSpec(rows, cols)  # a grid layout to place supports within a figure, (no rows, no cols)
        fig = plt.figure()

        for idx in range(len(self.species)):
            ax = fig.add_subplot(gs[idx])

            im = ax.imshow(self.species[idx].get_U(), interpolation="none", cmap=cm.viridis, vmin=0)
            ax.set_title(self.species[idx].get_name())

            divider = make_axes_locatable(ax) # Separate axes to be able to be moved
            cax = divider.append_axes("right", size="5%", pad=0.05) # moves axis to right side
            fig.colorbar(im, cax=cax, shrink=0.8)  # adds colour bar to plot

        fig.savefig('fig_setup.png')
        fig.show()
