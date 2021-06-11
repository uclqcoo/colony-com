from load_liquid_data import load_liquid_data
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import  curve_fit
from scipy.integrate import solve_ivp
import pyswarms as ps
import numpy as np



    #plt.plot(np.array(BP_data[IPTG_conc]).T, '--')
#plt.show()

def dGFP(t, GFP, IPTG_conc, params):


    #cell = cell_conc(t)

    # FOR NOW DONT INCLUDE NUTRIENTS

    #return hill_BP(IPTG_conc, na, ka, nr, kr, l, h) #* cell
    return hill_BP(IPTG_conc, params) #* cell

def hill_BP(conc, params):
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


def timeseries(IPTG_conc, t_samp,params):

    sim_ivp = solve_ivp(dGFP, [0, t_final], [0], t_eval=t_samp, args = (IPTG_conc, params))

    return sim_ivp.y


def all_timeseries(params):
    #na, ka, nr, kr, l, h = params
    #na, ka, l, h = params
    all_ts = []
    for IPTG_conc in IPTG_concs:

        ts = timeseries(IPTG_conc, t_samp, params)
        all_ts.append(ts)

    return all_ts

def f(x, na, ka, nr, kr, l, h):
    #IPTG_concs, t = x
    y_sim = []
    for conc in IPTG_concs:
        sim = timeseries(conc, t_samp, na, ka,  l, h)
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


filepath = '/home/neythen/Desktop/Projects/colony-com/IPTG_characterisation/data/liquid_culture/20191204_ZBD_first-test_16h_parsed_processed.csv'
n_points = 49

TH_data, BP_data = load_liquid_data(filepath, n_points)

data = BP_data

IPTG_concs = data.keys() #uM

print('IPTG concs:', IPTG_concs)

for IPTG_conc in IPTG_concs:
    d = np.array(data[IPTG_conc])
    d -= d[:, 0].reshape(-1, 1)
    data[IPTG_conc] = d
    #plt.figure()


t_final = 20 * n_points # mins threshold

# t_final = 20 * 47  # mins
dt = .1
t_points = int(t_final / dt)

t_eval = np.arange(0, t_final, dt)
t_samp = np.arange(0, t_final, 20)

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



popt, pcov = curve_fit(f, x_data, y_data, bounds = ([1, -7, 1, -7, -9, -3], [6, 3, 20, 3, 2, 2]),  p0 = [1.5, 0.12,10 , -2, -4, 0.5])

print(popt)
'''

options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9, 'k': 2, 'p': 10}
bounds = ([1, -7, 1, -7, -9, -3], [6, 3, 20, 3, 7, 7])   #na, ka, nr, kr, min, max
bounds = ([1, -7,-9, -3], [6, 3, 7, 7])   #na, ka, nr, kr, min, max
# Call instance of PSO
#optimizer = ps.single.GlobalBestPSO(n_particles=200, dimensions=4, options=options, bounds=bounds)

# Perform optimization
#cost, pos = optimizer.optimize(vector_objective, iters=1000, verbose=True)

#print(cost, pos)
pos = [ 2.60754348, -1.8366539,   4.75478292, -2.16703204,  0.34129256,  3.15083919] #95773182825.73466
#pos = [ 2.46076584, -2.30122155,  1.09231373,  2.42955927] #9666719050837.92
popt  = pos

sim_data = all_timeseries(pos)
sim_data = np.array(sim_data)
print(np.array(sim_data).shape)

all_data = []

for IPTG_conc in data.keys():
    all_data.append(data[IPTG_conc])

print(np.array(all_data).shape)

all_data = np.array(all_data)
IPTG_concs = list(BP_data.keys())

for i,time in enumerate([12, 18, 24, 30, 36, 48]):
    c = plt.rcParams['axes.prop_cycle'].by_key()['color'][i]
    print(time)
    plt.plot(np.array(IPTG_concs) / 1e6, np.mean(all_data[:, :, time], axis=1), color = c, label = str(int(time/3)) + 'h')
    plt.plot(np.array(IPTG_concs) / 1e6, sim_data[:, :, time], '--', color = c)
    plt.xscale('log')
plt.xlabel('log(IPTG), M')
plt.ylabel('calibrated GFP / calibrated OD, AU')
plt.legend()

plt.figure()
plt.plot(np.arange(50)/1e6, [hill_BP(conc, popt) for conc in range(50)])
plt.xscale('log')
plt.xlabel('log(IPTG), M')
plt.ylabel('rate of GFP production, AU/min')

plt.show()