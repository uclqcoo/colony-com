import numpy as np
import matplotlib.pyplot as plt


def load_liquid_data(filepath, n_points):
    BP_data = {}
    TH_data = {}

    with open(filepath) as file:
        file.readline()
        for i, line in enumerate(file):
            line = line.split(',')


            name = line[0]
            IPTG_conc = line[8] #uM
            replicate = line[9]
            time = line[11]
            flouresence = line[-1]
            OD = line[-2]




            if name == '"ZBD"':
                IPTG_conc = float(IPTG_conc)
                flouresence = float(flouresence)
                OD = float(OD)
                try:
                    BP_data[IPTG_conc].append(flouresence/OD)
                except:
                    BP_data[IPTG_conc] = []
                    BP_data[IPTG_conc].append(flouresence/OD)

            if name == '"ZG"':
                IPTG_conc = float(IPTG_conc)
                flouresence = float(flouresence)
                OD = float(OD)
                try:
                    TH_data[IPTG_conc].append(flouresence/OD)
                except:
                    TH_data[IPTG_conc] = []
                    TH_data[IPTG_conc].append(flouresence/OD)
    print(len(TH_data[list(TH_data.keys())[0]]))
    for IPTG_conc in TH_data.keys():
        timecourses = TH_data[IPTG_conc]

        # split the different repeats up

        repeats = []
        i = 0

        while (i + 1) * n_points <= len(timecourses):
            repeat = timecourses[i * n_points:(i + 1) * n_points]
            repeats.append(repeat)

            i += 1

        TH_data[IPTG_conc] = repeats

    for IPTG_conc in BP_data.keys():
        timecourses = BP_data[IPTG_conc]

        # split the different repeats up

        repeats = []
        i = 0
        while (i + 1) * n_points <= len(timecourses):
            repeat = timecourses[i * n_points:(i + 1) * n_points]
            repeats.append(repeat)

            i += 1

        BP_data[IPTG_conc] = repeats

    return TH_data, BP_data


if __name__ == '__main__':

    filepath= '/home/neythen/Desktop/Projects/colony-com/IPTG_characterisation/data/liquid_culture/20191204_ZBD_first-test_16h_parsed_processed.csv'
    n_points = 49

    TH_data, BP_data = load_liquid_data(filepath, n_points)

    print(TH_data.keys())

    #(concs, repeats, timepoints)

    all_data = []

    for IPTG_conc in TH_data.keys():
        all_data.append(TH_data[IPTG_conc])

    print(np.array(all_data).shape)

    all_data = np.array(all_data)
    IPTG_concs = list(TH_data.keys())

    for time in range(0, 49, 6):
        print(time)
        plt.plot(np.array(IPTG_concs)/1e6, np.mean(all_data[:,:,time], axis = 1))
        plt.xscale('log')



    plt.figure()

    print(TH_data.keys())

    # (concs, repeats, timepoints)

    all_data = []

    for IPTG_conc in BP_data.keys():
        all_data.append(BP_data[IPTG_conc])

    print(np.array(all_data).shape)

    all_data = np.array(all_data)
    IPTG_concs = list(BP_data.keys())

    for time in range(0, 49, 6):
        print(time)
        plt.plot(np.array(IPTG_concs) / 1e6, np.mean(all_data[:, :, time], axis=1))
        plt.xscale('log')

    plt.show()




