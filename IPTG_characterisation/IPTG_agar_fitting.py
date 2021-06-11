import sys
import os

import pyswarms as ps
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from load_const_data import load_const_data
from growth_fitting import gompertz


from scipy.optimize import  curve_fit
from scipy.integrate import solve_ivp

import numpy as np


def dGFP(t, GFP, IPTG_conc, params):


    cell = cell_conc(t)

    # FOR NOW DONT INCLUDE NUTRIENTS

    return hill_TH(IPTG_conc, params) * cell

def hill_BP(conc,params):

    na, ka, nr, kr, l, h = params
    conc /= 1000  # convert from uM to mM to match spot experiments
    ka = 10 ** ka
    kr = 10 ** kr
    l = 10 ** l

    h = 10 ** h
    #if conc < 0: conc = 0
    #conc[conc < 0] = 0


    act = conc**na/(ka**na + conc**na)
    rep = kr**nr/(kr**nr + conc**nr)

    return l + (h-l)*act*rep

def hill_TH(conc, params):
    na, ka, l, h = params
    conc /= 1000  # convert from uM to mM to match spot experiments
    ka = 10 ** ka

    l = 10 ** l

    h = 10 ** h
    #if conc < 0: conc = 0
    #conc[conc < 0] = 0


    act = conc**na/(ka**na + conc**na)


    return l + (h-l)*act


def timeseries(IPTG_conc, t_samp, params):

    sim_ivp = solve_ivp(dGFP, [0, t_final], [0], t_eval=t_samp, args = (IPTG_conc, params))

    return sim_ivp.y


def all_timeseries(params):

    all_ts = []
    for IPTG_conc in IPTG_concs:

        ts = timeseries(IPTG_conc, t_samp, params)
        all_ts.append(ts)

    return all_ts

def f(x, na, ka, nr, kr, l, h):
    #IPTG_concs, t = x
    y_sim = []
    for conc in IPTG_concs:
        sim = timeseries(conc, t_samp, na, ka, nr, kr, l, h)
        y_sim.append(sim)
    return np.hstack(y_sim)[0]


def vector_objective(params, *args):
    '''
    objective in vector form for the PSO library
    :param params:
    :param args:
    :return:
    '''


    errors = []
    #print(params)

    for i in range(len(params)):
        all_sim_data = all_timeseries(params[i,:])
        error = 0

        for i, IPTG_conc in enumerate(IPTG_concs):



            lab_data = np.array(np.array(data[IPTG_conc]))
            sim_data = np.array(all_sim_data[i])


            diff = lab_data-sim_data

            #error += np.sum(((diff)/(lab_data+0.00001))**2)/len(sim_data)
            error += np.sum(((diff))**2)/len(sim_data)


        with open('particle_swarm1.csv', 'a') as file:
            #print('error:', error, 'params:', params[i,:])
            file.write(str(error) + ',' + ','.join([str(p) for p in params[i,:]]) + '\n')

        errors.append(error)

    return errors

filepath= '/home/neythen/Desktop/Projects/synbiobrain/IPTG_characterisation/data/201202_IPTGagar_img_data_summary.csv'
n_points = 64
gompertz_ps = [1.24927088e-01, 1.79595968e-04, 3.48051876e+02] #bandpass
gompertz_ps = [1.34750462e-01, 1.90257947e-04, 3.33841052e+02] #threshold

cell_conc = lambda t: gompertz(t,*gompertz_ps)

TH_data, BP_data = load_const_data(filepath, n_points)

print(TH_data.keys())

data = TH_data
TH_concs = [0., 1., 2.5, 5., 10.]
BP_concs = [0., 1., 2.5, 5., 10., 50.]

IPTG_concs = TH_concs
for IPTG_conc in IPTG_concs:
    d = np.array(data[IPTG_conc])
    d -= d[:, 0].reshape(-1, 1)
    data[IPTG_conc] = d
    #plt.figure()



    #plt.plot(np.array(BP_data[IPTG_conc]).T, '--')
#plt.show()

t_final = 20 * 64  # mins threshold

# t_final = 20 * 47  # mins
dt = .1
t_points = int(t_final / dt)

t_eval = np.arange(0, t_final, dt)
t_samp = np.arange(0, t_final, 20)
IPTG_concs = [0., 1., 2.5, 5., 10., 50.]

options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9, 'k': 2, 'p': 10}
bounds = ([1, -7, 1, -7, -9, -3], [6, 3, 20, 3, 2, 2])   #na, ka, nr, kr, min, max
bounds = ([1, -10,  -9, -9], [10, 3, 3, 3])   #n, k,  min, max
# Call instance of PSO
#optimizer = ps.single.GlobalBestPSO(n_particles=100, dimensions=4, options=options, bounds=bounds)

# Perform optimization
#cost, pos = optimizer.optimize(vector_objective, iters=1000, verbose=True)
#print(cost, pos)
'''
# construct x data
x = []
for conc in IPTG_concs:
    concs = [conc]*len(t_samp)
    x.append(np.vstack((concs, t_samp)))

x_data = np.hstack(x)

y_data = []

for c in IPTG_concs:

    y_data.append(np.mean(BP_data[c], axis = 0))
y_data = np.hstack(y_data).T
print(y_data.shape)

#print(timeseries([10, 90], 1,1,1,1,1,1))


popt, pcov = curve_fit(f, x_data, y_data, bounds = ([1, -8, 1, -8, -3, -3], [6, 2, 6, 2, 2, 2]),  p0 = [3.5, -3, 3.5, -3, -1, -1])

print(popt)
'''


params = [ 1.58646428, -0.6564445,   7.99974394, -2.14317698, -3.88586878, -0.37292162] #0.010032104093890572

#params = [ 1.47076973,  0.12880582, 19.96940403, -2.05425021, -3.9020145,   0.56923747] #0.009755106363584246

params =  [ 9.99987666, -2.28022186, -3.34040721, -2.18816125] #4.135360430175275








concs = np.linspace(0,50, 50)

plt.plot(np.linspace(0,50/1e6, 50), hill_TH(concs, params))
plt.xlabel('IPTG (M)')
plt.xscale('log')
plt.ylabel('Rate of GFP production (MPF/min)')


plt.figure()
for i, IPTG_conc in enumerate([0., 1., 2.5, 5., 10.]):

    plt.subplot(2,3,i+1)
    inf = timeseries(IPTG_conc, t_eval,params)
    print(inf[0][-1])
    print(inf.shape)
    plt.plot(t_eval, inf[0])
    plt.title(str(IPTG_conc) + 'uM IPTG')
    plt.ylabel('Mean pixel flourescence')
    plt.xlabel('Time (mins)')
    plt.ylim(top=0.6)
    plt.plot(t_samp, np.array(data[IPTG_conc]).T, '--')
plt.show()


