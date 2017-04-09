'''
Created on Mar 26, 2017

@author: jjordening
'''
import glob
import argparse
import pandas as pd
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'path',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    args = parser.parse_args()
    df = pd.read_csv(args.path)
    df['speed'] = np.sqrt(df['vel0']**2+df['vel1']**2+df['vel2']**2)
    df.to_csv(args.path[:-4]+'New.txt', index = False)
    
        