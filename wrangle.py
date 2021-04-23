import numpy as np
import pandas as pd
import math
from sklearn import metrics
import datetime

from scipy.stats import entropy

import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates #to format dates on our plots

import seaborn as sns

'''
*------------------*
|                  |
|     WRANGLE      |
|                  |
*------------------*
'''

def get_logs():
    """
    get_logs reads in txt file to csv
    - renames cols
     - reassigns dtypes
     - merges data-time and sets as index
     - fills nulls
    """
    headers = ['col1', 'col2', 'page', 'userid', 'cohort', 'ip']
    dtypes = {'col1': 'str', 'col2': 'str', 'col3': 'str', 'col4': 'int', 'col5': 'float', 'col6': 'str'}
    
    df = pd.read_csv("anonymized-curriculum-access.txt", sep=' ', header=None, names=headers, dtype=dtypes)

    df['date'] = df['col1']+ " " + df['col2']
    
    df.date = pd.to_datetime(df.date)
    
    df = df.drop(columns=['col1', 'col2']).set_index('date').sort_index()
    
    df['cohort'] = np.where(df.cohort.isnull(), 0, df.cohort)
    
    return df




'''
*------------------*
|                  |
|     EXPLORE      |
|                  |
*------------------*
'''

def get_lower_and_upper_bounds(df, feature, m = 1.5):
    '''
    get_lower_and_upper_bounds will take in a dataframe's feature:
    - calculate it's 1st & 3rd quartiles,
    - use their difference to calculate the IQR
    - then apply to calculate upper and lower bounds
    - using the `m` multiplier
    - default multiplier is '1.5'
    '''
    q1 = df[feature].quantile(.25)
    q3 = df[feature].quantile(.75)
    
    iqr = q3 - q1
    
    upper_bound = q3 + (m * iqr)
    lower_bound = q1 - (m * iqr)
    
    return upper_bound, lower_bound