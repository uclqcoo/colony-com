import os
from load_spot_data import load_spot_data
import numpy as np
import matplotlib.pyplot as plt

filepath_growth_TH = '/home/neythen/Desktop/Projects/colony-com/IPTG_characterisation/data/growth/201201_IPTGsendersZG_ODimg_data_summary.csv'
colours = ['b', 'red', 'g']
n_points = 62
OD_data = load_spot_data(filepath_growth_TH, n_points)


filepath_TH = os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'), '201201_IPTGsendersZG_img_data_summary.csv')
n_points = 62

GFP_data = load_spot_data(filepath_TH, n_points)

for IPTG_conc in [0., 5., 10.]:
    for distance in [4.5, 9.0, 13.5]:
        plt.figure()
        GFP = np.array(GFP_data[IPTG_conc][distance]).T
        OD = np.array(OD_data[IPTG_conc][distance]).T

        plt.plot(OD, GFP)

plt.show()