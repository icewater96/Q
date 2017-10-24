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

config_file_path = r'..\..\L_C config.txt'


def load_config(file_path):
    with open(file_path, 'r') as f:
        api_id = f.readline().strip()
        account_id = f.readline().strip()
    return api_id, account_id

def get_listed_loans():
    api_id, _ = load_config(config_file_path)
    header = {'Authorization' : api_id,
              #'Content-Type': 'application/json', 
              #'Accept': 'application/json', 
              "X-LC-LISTING-VERSION":"1.3"
              }
    resp = requests.get("https://api.lendingclub.com/api/investor/v1/loans/listing", 
                        headers= header, params = {'showAll': 'true'})
    
    loan_df = pd.DataFrame(resp.json()['loans'])
    resp.close()
    
    print(loan_df.shape)
    
    return loan_df

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