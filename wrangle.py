import numpy as np
import pandas as pd
import math
from sklearn import metrics
import datetime

from env import host, user, password

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





def get_connection(db, user=user, host=host, password=password):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    '''
    
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'



    

def acq_logs():
    """
    This function pulls in a full JOINed dataset from codeup's db
    And returns a pandas DF.
    """
    sql_query = """
    SELECT *
    FROM logs as l
    JOIN cohorts as c ON c.id = l.cohort_id
    """
    return pd.read_sql(sql_query, get_connection('curriculum_logs'))





def prep_logs():
    """
    prep_logs reads in SQL
    - renames cols
     - reassigns dtypes
     - merges data-time and sets as index
     - adds day/month col
     - rename observations in program_id
    """
    df = acq_logs()

    df['datetime'] = df['date']+ " " + df['time']
    
    df.datetime = pd.to_datetime(df.datetime)
    
    df = df.drop(columns=['date', 'time', 'id', 'slack', 'created_at', 'updated_at', 'deleted_at']).set_index('datetime').sort_index()
    
    df['weekday'] = df.index.day_name()
    
    df['month'] = df.index.month
    
    df['program_id'] = df.program_id.replace({1: "FS_PHP", 2: "FS_Java", 3: "DS", 4: "frontend"})
    
    return df





def zillow17():
    '''
    This function reads in the zillow data from the Codeup db
    and returns a pandas DataFrame with:
    - all fields related to the properties that are available
    - using all the tables in the database
    - Only include properties with a transaction in 2017
    - include only the last transaction for each property
    - zestimate error
    - date of transaction
    - Only include properties that include a latitude and longitude value
    '''
    
    query = """
            SELECT prop.*,
                   pred.logerror,
                   pred.transactiondate,
                   air.airconditioningdesc,
                   arch.architecturalstyledesc,
                   build.buildingclassdesc,
                   heat.heatingorsystemdesc,
                   landuse.propertylandusedesc,
                   story.storydesc,
                   construct.typeconstructiondesc
            FROM   properties_2017 prop
            INNER JOIN (SELECT parcelid,
                               logerror,
                               Max(transactiondate) transactiondate
                        FROM   predictions_2017
                        GROUP  BY parcelid, logerror) pred
                     USING (parcelid)
            LEFT JOIN airconditioningtype air USING (airconditioningtypeid)
            LEFT JOIN architecturalstyletype arch USING (architecturalstyletypeid)
            LEFT JOIN buildingclasstype build USING (buildingclasstypeid)
            LEFT JOIN heatingorsystemtype heat USING (heatingorsystemtypeid)
            LEFT JOIN propertylandusetype landuse USING (propertylandusetypeid)
            LEFT JOIN storytype story USING (storytypeid)
            LEFT JOIN typeconstructiontype construct USING (typeconstructiontypeid)
            WHERE prop.latitude IS NOT NULL
                  AND prop.longitude IS NOT NULL
                  AND transactiondate like '2017%%'
    """
    
    return pd.read_sql(query, get_connection('zillow'))




'''
*------------------*
|                  |
|     PREPARE      |
|                  |
*------------------*
'''
    
def drop_based_on_pct(df, pc, pr):
    """
    drop_based_on_pct takes in: 
    - dataframe, 
    - threshold percent of non-null values for columns(# between 0-1), 
    - threshold percent of non-null values for rows(# between 0-1)
    Returns: a dataframe with the columns and rows dropped as indicated.
    """
    tpc = 1-pc
    tpr = 1-pr
    df.dropna(axis = 1, thresh = tpc * len(df.index), inplace = True)
    df.dropna(axis = 0, thresh = tpr * len(df.columns), inplace = True)
    return df
    
    


    
def outlier(df, feature, m):
    '''
    outlier will take in a dataframe's feature:
    - calculate it's 1st & 3rd quartiles,
    - use their difference to calculate the IQR
    - then apply to calculate upper and lower bounds
    - using the `m` multiplier
    '''
    q1 = df[feature].quantile(.25)
    q3 = df[feature].quantile(.75)
    
    iqr = q3 - q1
    
    multiplier = m
    upper_bound = q3 + (multiplier * iqr)
    lower_bound = q1 - (multiplier * iqr)
    
    return upper_bound, lower_bound




    
    
def clean_zillow(df):
    """
    clean_zillow will:
    - read in df acquired from SQL query
    - filter data to single unit homes with min 1B/1B over 500 sqft
    - drop columns with 40%+ & rows 30%+ null
    - add a column for county names
    - drop unnecessary columns
    - fills in unitcnt/lotsizesquarefeet/buildingqualitytypeid nulls
    - drops remaining nulls
    - removes extreme outliers for home_value & sqft
    - adds a home_age & log_quartiles column
    - converts certain float columns to int
    - renames certain columns
    """
    
    #df = pd.read_csv('zillow.csv')
    #df = df.set_index("parcelid")
    
    # Restrict df to only properties that meet single-use criteria
    single_use = [260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 273, 275, 276, 279]
    df = df[df.propertylandusetypeid.isin(single_use)]
    
    # Filter those properties without at least 1 bath & bed and 500 sqft area
    df = df[(df.bedroomcnt > 0) & (df.bathroomcnt > 0) & ((df.unitcnt<=1)|df.unitcnt.isnull())\
            & (df.calculatedfinishedsquarefeet>500)]
    
    # Handle missing values i.e. drop columns and rows based on a threshold
    df = drop_based_on_pct(df, .6, .7)
    
    # Add column for counties
    df['county'] = np.where(df.fips == 6037, 'Los_Angeles',
                           np.where(df.fips == 6059, 'Orange',
                                   'Ventura'))
    
    # Drop unnecessary/redundant columns
    df = df.drop(['id', 'calculatedbathnbr', 'finishedsquarefeet12', 
                  'fullbathcnt', 'heatingorsystemtypeid', 
                  'propertycountylandusecode', 'propertylandusetypeid', 
                  'propertyzoningdesc', 'censustractandblock', 'propertylandusedesc', 
                  'heatingorsystemdesc', 'assessmentyear', 'regionidcounty' ],axis=1)
    
    
    # Replace nulls with median values for select columns
    df.lotsizesquarefeet.fillna(7265, inplace = True)
    df.buildingqualitytypeid.fillna(7.0, inplace = True)
    
    # Drop any remaining nulls
    df = df.dropna()
    
    # Columns that need to be adjusted for outliers
    df = df[df.taxvaluedollarcnt < 5_000_000]
    df = df[df.calculatedfinishedsquarefeet < 12500]
    
    # create column for age of home
    df['home_age'] = 2021 - df.yearbuilt
    
    # List of cols to convert to 'int'
    cols = ['fips', 'buildingqualitytypeid', 'bedroomcnt', 
            'home_age', 'yearbuilt']
    # loop through cols list in conversion
    for col in cols:
        df[col] = df[col].astype('int')
        
    # Rename columns
    df.rename(columns={"bathroomcnt": "bathrooms", 
                   "bedroomcnt": "bedrooms",
                   "buildingqualitytypeid": "property_quality", 
                   "calculatedfinishedsquarefeet": "sqft",
                   "lotsizesquarefeet": "lot_sqft",
                   "regionidzip": "zip_code",
                   "landtaxvaluedollarcnt": "land_value",
                   "structuretaxvaluedollarcnt": "structure_value",
                   "taxvaluedollarcnt": "home_value"
                  }, inplace=True)
    

    # create a categorical version of target by splitting into quartiles
    df['logerror_quartiles'] = pd.qcut(df.logerror, q=4, labels=['q1', 'q2', 'q3', 'q4'])
    
    # Drop unnecessary/redundant columns
    df = df.drop(['rawcensustractandblock', 'regionidcity', 'zip_code', 'roomcnt', 'unitcnt', 'transactiondate'],axis=1)
    
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