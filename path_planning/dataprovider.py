import numpy as np
import csv
import os

def load_simulator_data(csvfname):
    """
    Load dataset from csv file
    """
    data=[]
    with open(csvfname, 'r') as csvfile:
        data_tmp = list(csv.reader(csvfile, delimiter=','))
        for row in data_tmp:
            x7=[float(x) for x in row[7].split(':')]
            x8=[float(x) for x in row[8].split(':')]
            
            data.append(((row[0],row[1],row[2]),np.array([float(row[3]),float(row[4]),float(row[5]),float(row[6])]+x7+x8)))

    return data