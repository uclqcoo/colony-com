import numpy as np
import matplotlib.pyplot as plt
import copy
import math
from load_spot_data import *
from scipy.optimize import  curve_fit

def gompertz(t, A, um, lam):


    return A* np.exp(-np.exp((um*np.e)/A*(lam - t) +1))


if __name__ == '__main__':

    filepath_growth_TH = '/home/neythen/Desktop/Projects/colony-com/IPTG_characterisation/data/growth/201201_IPTGsendersZG_ODimg_data_summary.csv'
    n_points = 62
    filepath_growth_BP = '/home/neythen/Desktop/Projects/colony-com/IPTG_characterisation/data/growth/201124_IPTGsendersZBD_ODimg_data_summary.csv'
    n_points = 66
    colours = ['b', 'red', 'g']

    #growth_data_TH = load_spot_data(filepath_growth_TH, n_points)
    growth_data_BP = load_spot_data(filepath_growth_BP, n_points)

    normalised_growth_data = copy.deepcopy(growth_data_BP)

    all_normed_data = []
    for IPTG_conc in [0., 5., 10., 50., 100., 500.]:
        plt.figure()
        plt.title('Threshold: ' + str(IPTG_conc))
        for i, distance in enumerate([4.5, 9.0, 13.5]):
            print(np.min(normalised_growth_data[IPTG_conc][distance]))
            normalised_growth_data[IPTG_conc][distance] = np.array(growth_data_BP[IPTG_conc][distance]).T - np.array(growth_data_BP[IPTG_conc][distance])[:,0]# + 0.136383442265795
            print(normalised_growth_data[IPTG_conc][distance].shape)
            all_normed_data.extend(normalised_growth_data[IPTG_conc][distance].T)
            #all_normed_data.extend(normalised_growth_data[IPTG_conc][distance]/normalised_growth_data[IPTG_conc][distance][0,:])
            #plt.plot(np.log(normalised_growth_data[IPTG_conc][distance]/normalised_growth_data[IPTG_conc][distance][0,:]), colours[i])
            plt.plot(np.linspace(0, n_points*20, n_points), normalised_growth_data[IPTG_conc][distance], colours[i])
    #plt.show()
    all_normed_data = np.array(all_normed_data)
    all_normed_data = np.mean(all_normed_data, axis = 0)
    print(all_normed_data.shape)




    time_points = np.linspace(0, n_points*20, n_points)# min
    popt,pcov = curve_fit(gompertz, time_points, all_normed_data, p0 = (0.28, 0.001, 300))

    print(popt)

    fitted_data = gompertz(time_points, popt[0], popt[1], popt[2])

    plt.figure()
    plt.plot(time_points, fitted_data)
    plt.plot(time_points, all_normed_data)
    plt.show()