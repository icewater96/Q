# -*- coding: utf-8 -*-
"""
Created on Tue Nov 08 23:51:03 2016

@author: l96
"""
import pandas as pd
import numpy as np

import sys
sys.path.append('..\..\Library')
import explore_raw_data

#%% Notes
# 1) joint and separate
# 2) 36 months vs 60 months
# 3) 

#%% 
filename_pool = [r'Data\LoanStats3a - trimmed.csv',
                 r'Data\LoanStats3b - trimmed.csv',
                 r'Data\LoanStats3c - trimmed.csv', 
                 r'Data\LoanStats3d - trimmed.csv', 
                 r'Data\LoanStats_2016Q1 - trimmed.csv', 
                 r'Data\LoanStats_2016Q2 - trimmed.csv', 
                 r'Data\LoanStats_securev1_2016Q3 - trimmed.csv']

temp_df_pool = []                 
for filename in filename_pool:
    temp = pd.read_csv(filename, header = 0, low_memory = False, encoding = 'utf-8' )
    print temp.shape
    temp_df_pool.append(temp)
    
raw_data = pd.concat(temp_df_pool, axis = 0, ignore_index = True)    
    
raw_data_description, columns_for_export = explore_raw_data.explore_columns(raw_data)

# Column description
column_description = pd.read_excel('Data\LCDataDictionary.xlsx')
column_description.rename(columns = {'LoanStatNew': 'Column Name'}, inplace = True)

all_description = pd.merge(column_description, raw_data_description, left_on = 'Column Name', 
                       right_on = 'Column Name', how = 'outer' )

#%% Categorize columns    
columns_to_remove = ['url', 'verification_status_joint', 'dti_joing', 'annual_inc_joint', 
                     'fico_range_high', 'fico_range_low', 'last_fico_range_high', 
                     'last_fico_range_low', 'policy_code', ]
                    
columns_to_keep = ['member_id', 'load_amnt', 'dti', ]           

all_description['Parsing Action'] = '' * len(all_description)
for column in columns_to_remove:
    all_description.ix[all_description['Column Name'] == column, 'Parsing Action'] = 'Remove'
    
for column in columns_to_keep:
    all_description.ix[all_description['Column Name'] == column, 'Parsing Action'] = 'Keep'
                     

if False:
    all_description.to_csv('a.csv', encoding = 'utf-8', columns = ['Parsing Action', 'Description'] + columns_for_export)



