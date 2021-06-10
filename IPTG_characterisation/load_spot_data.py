import numpy as np
import matplotlib.pyplot as plt


def load_spot_data(filepath, n_points):
    # dictionary structure: time_courses = dict[spot_concentration][distance]
    observed_wells = []  # to keep track of when we start a new repeat

    data = {}
    with open(filepath) as file:
        file.readline()
        for i, line in enumerate(file):
            line = line.split(',')

            time_point = int(line[1]) * 20.0  # each timepoint is 20mins
            flouresence = float(line[4])
            IPTG_conc = float(line[-2])
            distance = float(line[-1])
            n = line[-3]

            try:
                data[IPTG_conc][distance].append(flouresence)
            except:
                try:

                    data[IPTG_conc][distance] = []
                    data[IPTG_conc][distance].append(flouresence)
                except:

                    data[IPTG_conc] = {}
                    data[IPTG_conc][distance] = []
                    data[IPTG_conc][distance].append(flouresence)

    # print(data)
    for IPTG_conc in data.keys():


        for d, distance in enumerate(data[IPTG_conc].keys()):
            timecourses = data[IPTG_conc][distance]

            # split the different repeats up

            repeats = []
            i = 0

            while (i + 1) * n_points <= len(timecourses):


                repeat = timecourses[i * n_points:(i + 1) * n_points]
                repeats.append(repeat)

                i += 1

            data[IPTG_conc][distance] = repeats
    return data


def load_spot_data_small_concs(filepath, n_points, activation_func = '"ZG"'):
    # dictionary structure: time_courses = dict[spot_concentration][distance]
    observed_wells = []  # to keep track of when we start a new repeat

    data = {}
    with open(filepath) as file:
        file.readline()
        for i, line in enumerate(file):
            line = line.split(',')

            time_point = int(line[1]) * 20.0  # each timepoint is 20mins
            flouresence = float(line[18])
            IPTG_conc = float(line[14])
            distance = float(line[13])
            n = str(line[11])


            if n == activation_func:

                try:
                    data[IPTG_conc][distance].append(flouresence)
                except:
                    try:

                        data[IPTG_conc][distance] = []
                        data[IPTG_conc][distance].append(flouresence)
                    except:

                        data[IPTG_conc] = {}
                        data[IPTG_conc][distance] = []
                        data[IPTG_conc][distance].append(flouresence)

    # print(data)
    for IPTG_conc in data.keys():

        for d, distance in enumerate(data[IPTG_conc].keys()):
            timecourses = data[IPTG_conc][distance]

            # split the different repeats up

            repeats = []
            i = 0

            while (i + 1) * n_points <= len(timecourses):
                repeat = timecourses[i * n_points:(i + 1) * n_points]
                repeats.append(repeat)

                i += 1

            data[IPTG_conc][distance] = repeats
    return data

if __name__ == '__main__':
    colours = ['red', 'g', 'b']
    '''
    filepath_BP = '/home/neythen/Desktop/Projects/synbiobrain/IPTG_characterisation/data/201124_IPTGsendersZBD_img_data_summary.csv'
    n_points = 67

    data_BP = load_spot_data(filepath_BP, n_points)
    for IPTG_conc in [0., 5., 10., 50., 100., 500.]:
        plt.figure()
        plt.title('Bandpass: ' + str(IPTG_conc))
        for i,distance in enumerate([4.5, 9.0, 13.5]):
            plt.plot(np.array(data_BP[IPTG_conc][distance]).T, colours[i])



    filepath_TH = '/home/neythen/Desktop/Projects/synbiobrain/IPTG_characterisation/data/201201_IPTGsendersZG_img_data_summary.csv'
    n_points = 62
    data_TH = load_spot_data(filepath_TH, n_points)
    for IPTG_conc in [0., 5., 10., 50., 100., 500.]:
        plt.figure()
        plt.title('Threshold: ' + str(IPTG_conc))
        for i,distance in enumerate([4.5, 9.0, 13.5]):
            plt.plot(np.array(data_TH[IPTG_conc][distance]).T, colours[i])
    plt.show()
    

    filepath_growth_TH = '/home/neythen/Desktop/Projects/colony-com/IPTG_characterisation/data/growth/201201_IPTGsendersZG_ODimg_data_summary.csv'

    n_points = 62
    growth_data_TH = load_spot_data(filepath_growth_TH, n_points)
    for IPTG_conc in [0., 5., 10., 50., 100., 500.]:
        plt.figure()
        plt.title('Threshold: ' + str(IPTG_conc))
        for i,distance in enumerate([4.5, 9.0, 13.5]):
            plt.plot(np.array(growth_data_TH[IPTG_conc][distance]).T, colours[i])
    plt.show()
    

    filepath_growth_BP = '/home/neythen/Desktop/Projects/colony-com/IPTG_characterisation/data/growth/201124_IPTGsendersZBD_ODimg_data_summary.csv'

    n_points = 66
    growth_data_BP = load_spot_data(filepath_growth_BP, n_points)
    for IPTG_conc in [0., 5., 10., 50., 100., 500.]:
        plt.figure()
        plt.title('Bandpass: ' + str(IPTG_conc))
        for i, distance in enumerate([4.5, 9.0, 13.5]):
            plt.plot(np.array(growth_data_BP[IPTG_conc][distance]).T, colours[i])
    plt.show()
    

    filepath_small_concs = '/home/neythen/Desktop/Projects/colony-com/IPTG_characterisation/data/210114_IPTGspot_norm_data_summary.csv'

    n_points = 73
    GFP_small_concs = load_spot_data_small_concs(filepath_small_concs, n_points)
    for IPTG_conc in [0.0, 1.0, 2.5]:
        plt.figure()
        plt.title('Threshold' + str(IPTG_conc))
        for i, distance in enumerate([4.5, 9.0, 13.5]):
            plt.plot(np.array(GFP_small_concs[IPTG_conc][distance]).T, colours[i])
    plt.show()
    '''

    filepath_small_concs = '/home/neythen/Desktop/Projects/colony-com/IPTG_characterisation/data/210114_IPTGspot_norm_data_summary.csv'
    n_points = 73
    GFP_small_concs = load_spot_data_small_concs(filepath_small_concs, n_points, activation_func='"ZBD"')
    for IPTG_conc in [0.0, 1.0, 2.5]:
        plt.figure()
        plt.title('Threshold' + str(IPTG_conc))
        for i, distance in enumerate([4.5, 9.0, 13.5]):
            plt.plot(np.array(GFP_small_concs[IPTG_conc][distance]).T, colours[i])
    plt.show()