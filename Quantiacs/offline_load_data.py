# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 00:19:15 2017

@author: l96
"""

import pandas as pd
import numpy as np
import os
import zipfile

root_dir = r'tickerData'

filename_pool = os.listdir(root_dir)

def write_dict_to_hdf_and_zip(ticker_dict, hdf_filename, compress = True):
    # ticker_dict: key = ticker, value = DataFrame
    # Assume .h5 and .zip have same filename base
    
    with pd.HDFStore(hdf_filename, 'w') as f:
        for ticker_name, ticker_df in data_dict.items():
            f[ticker_name] = ticker_df
    
    if compress:
        temp = list(os.path.splitext(hdf_filename))
        zip_filename = temp[0] + r'.zip'
        
        with zipfile.ZipFile(zip_filename, 'w', compression = zipfile.ZIP_DEFLATED) as f:
            f.write(hdf_filename)

def read_hdf_or_zip_to_dict(filename):
    # Output 
    temp = os.path.splitext(filename)
    if temp[-1] == '.zip':
        with zipfile.ZipFile(filename, 'r') as z:
            with pd.HDFStore(z.namelist()[0], 'r') as hdf:
                returned = {}
                for key in hdf.keys():
                    # Because keys in .h5 starts with '/', need to remove it. 
                    returned[key[1:]] = hdf.get(key)
    else:
        with pd.HDFStore(filename, 'r') as hdf:
            returned = {}
            for key in hdf.keys():
                # Because keys in .h5 starts with '/', need to remove it. 
                returned[key[1:]] = hdf.get(key)
    return returned

#%% Load data
data_dict = {}
for filename in filename_pool:
    dot_position = filename.find('.')
    ticker = filename[:dot_position]
    print(ticker)
    
    temp_df = pd.read_csv(os.path.join(root_dir, filename), parse_dates = ['DATE'])
    
    data_dict[ticker] = temp_df
    
if False:
    write_dict_to_hdf_and_zip(data_dict, 'data.h5')