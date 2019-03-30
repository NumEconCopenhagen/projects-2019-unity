import pandas_datareader.data as web
import datetime
import pandas as pd
import numpy as np


companys=['GOOG','^OMXC20'] 
from_year = 2011
to_year = 2017

start_download = datetime.datetime((from_year-1), 1, 1)
start_show = datetime.datetime((from_year), 1, 1)
end = datetime.datetime(to_year, 12, 31)

    
    # Download data
d = web.DataReader(companys, 'yahoo', start_download, end)

    #rename columns Symbols to Company 
d.columns.rename(['Attributes','Company'],inplace  = True)
d.rename(columns = {'Adj Close': 'Adj_Close'} ,inplace  = True)
d.rename(columns = {'GOOG':'Google'}, inplace = True)
d

d['Adj Close'].plot()
d = d.swaplevel(axis=1).sort_index(axis=1)
d

d.interpolate(inplace = True)
d

    # make 200 and 50 days running mean :
rm_200 = d.rolling(200).mean()['Adj Close']
rm_50 = d.rolling(50).mean()['Adj Close']

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


d.swaplevel(axis=1)['Adj Close'].plot()