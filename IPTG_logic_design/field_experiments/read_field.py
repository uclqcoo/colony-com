import numpy as np
import matplotlib.pyplot as plt
def read_field(filepath):
    # rows A-P
    # cols 1 - 24
    # 68 timepoints
    time_course = np.zeros(( 16, 24, 62))

    with open(filepath) as file:
        file.readline()


        for line in file:
            line = line.split(',')
            t = int(line[1])
            row = int(ord(line[2][1])-65)
            col = int(line[3])-1
            flouresence = float(line[4])

            if t <=61:
                time_course[row, col, t] = flouresence

    return time_course




if __name__== '__main__':

    two_input_path = '/home/neythen/Desktop/Projects/colony-com/IPTG_logic_design/field_experiments/20210330_ZG-2input_img_data_summary.csv'
    right_input_path = '/home/neythen/Desktop/Projects/colony-com/IPTG_logic_design/field_experiments/20210401_ZG-Rinput_img_data_summary.csv'
    left_input_path = '/home/neythen/Desktop/Projects/colony-com/IPTG_logic_design/field_experiments/20210331_ZG-Linput_img_data_summary.csv'
    time_course = read_field(left_input_path)

    plt.imshow(time_course[:,:,-1])
    plt.show()