import pandas_datareader.data as web
import datetime
import pandas as pd
import numpy as np



def download_data_with_runmean(companys=['GOOG'], from_year = 2011 , to_year = 2017):
    
    # We download an extra year back so we have running mean for the whole period 
    # (we have to go more than 200 days back because of holidays etc.)

    start_download = datetime.datetime((from_year-1), 1, 1)
    start_show = datetime.datetime((from_year), 1, 1)
    end = datetime.datetime(to_year, 12, 31)

    
    # Download data
    d = web.DataReader(companys, 'yahoo', start_download, end)

    #rename columns Symbols to Company 
    d.columns.rename(['Attributes','Company'],inplace  = True)
    d.interpolate(inplace = True)

    # make 200 and 50 days running mean :
    rm_200 = d.rolling(200).mean()['Close']
    rm_50 = d.rolling(50).mean()['Close']

    ## these two new DataFrames are now concated to one with a multiIndex
    rm = pd.concat((rm_200,rm_50), keys=['rm_200', 'rm_50'], names=['Attributes', 'Company'], axis=1)

    # Which can then be merged to the main DataFrame, 
    # # using only join since both have same multiindex (in column) and index(Date)
    d = d.join(rm)


    # MultiIndexes have levels, right now Atributes is first, Symbols(company)is second
    # this line makes Symbos first 
    d = d.swaplevel(axis=1)
    #Sort DataFrame
    d.sort_index(axis=1,inplace=True)


    d = d.loc[start_show:]

    return d


