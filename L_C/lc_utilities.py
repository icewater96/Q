# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 22:43:54 2017

@author: l96
"""

import requests
import os
import sys
import pandas as pd
import numpy as np
import json
import time

# Set user's account info path.
if os.environ['COMPUTERNAME'] == 'WSUSMIAMJ03QVWU':
    config_file_path = r'D:\temp\Do Not Touch.txt'
else:
    config_file_path = r'..\..\L_C config.txt'


def load_config(file_path):
    with open(file_path, 'r') as f:
        api_id = f.readline().strip()
        account_id = f.readline().strip()
    return api_id, account_id

def get_listed_loans():
    '''Loans are listed on the Lending Club platform at 6AM, 10AM, 2PM, and 6PM every day.
       This should be California time. Eastern Time - 3 hours
    '''
    api_id, _ = load_config(config_file_path)
    header = {'Authorization' : api_id,
              #'Content-Type': 'application/json', 
              #'Accept': 'application/json', 
              "X-LC-LISTING-VERSION":"1.3"
              }
    resp = requests.get("https://api.lendingclub.com/api/investor/v1/loans/listing", 
                        headers= header, params = {'showAll': 'true'})
    json_dict = resp.json()
    loan_df = pd.DataFrame(json_dict['loans'])
    timestamp_string = json_dict['asOfDate']
    resp.close()
    
    print(loan_df.shape)
    
    return loan_df, timestamp_string

def get_account_summary():
    api_id, account_id = load_config(config_file_path)
    header = {'Authorization' : api_id,
              #'Content-Type': 'application/json', 
              #'Accept': 'application/json', 
              #"X-LC-LISTING-VERSION":"1.3"
              }
    resp = requests.get(r'https://api.lendingclub.com/api/investor/v1/accounts/' + account_id + r'/summary',
                        headers = header,
                        )
#    resp = requests.get(r'https://api.lendingclub.com/api/investor/v1/accounts/83102326/summary',
#                        #headers = header,
#                        )    
    summary_df = pd.Series(resp.json())
    resp.close()
    
    print(summary_df.shape)
    
    return summary_df
    
if __name__ == '__main__':
    df_list = []
    counter = 0
    while 1:
        counter += 1
        df, timestamp_string = get_listed_loans()
        df['counter'] = counter
        df['Download Time'] = timestamp_string
        df.to_csv(timestamp_string.replace(':', '_')[:19] + '.csv', encoding = 'utf-8')
        
        df_list.append(df)
        print('sleeping ...')
        time.sleep(60 * 30)
        print('wake up ...')