import pandas_datareader.data as web
import datetime
import pandas as pd
import numpy as np


# set timeframe we want to analyze:
start = datetime.datetime(2011, 1, 1)
end = datetime.datetime(2017, 12, 31)

### Data import ####

# import data
d = web.DataReader(['TSLA', 'GOOG', 'MAERSK-A.CO'], 'yahoo', start, end)
# set index to date
# not neacesary d.index = pd.to_datetime(d.index)
# add column indicating year notused
#d['year']=d.index.year

#rename columns Symbols to Company 
d.columns.rename(['Attributes','Company'],inplace  = True)

#show DataFrame
d

#make into tall format not used
''' 
d = d.T.stack().rename_axis(['Date','Company','Attributes']).reset_index()

d.columns
'''


# checks for missing values
np.where(pd.isnull(d))

### Data cleaning ###
'''
Because the Danish and American stockexchange are sometimes open on different days, 
data is missing for some days when one is open and the other is not
for these days the NaN (missing value) will be replaced with the mean of the two surrounding values
'''

# replaces value with mean of the two surrounding values:
d.interpolate(inplace = True)

'''
lternatively values could have been replaced with the value of the previous day
# replaces missing values with the previous day
d.fillna(method='ffill',inplace = True)
'''
# rechecks for missing values
np.where(pd.isnull(d))


### Creating aditional columns ###
 
# make 200 and 50 days running mean :
rm_200 = d.rolling(200).mean()['Close']
rm_50 = d.rolling(50).mean()['Close']

#checking for missing values 
# (x-coordinates should go no higher than 198 and 48, below this is accepted since the means needs 
np.where(pd.isnull(rm_200))
np.where(pd.isnull(rm_50))

## these two new DataFrames are now concated to one with a multiIndex

rm = pd.concat((rm_200,rm_50), keys=['rm_200', 'rm_50'], names=['Attributes', 'Company'], axis=1)

# Which can then be merged to the main DataFrame, 
# using only join since both have same multiindex (in column) and index(Date)
d = d.join(rm)

# check it out:
d


### extra clean-up
# MultiIndexes have levels, right now Atributes is first, Symbols(company)is second
# this line makes Symbos first 
d = d.swaplevel(axis=1)
#Sort DataFrame
d.sort_index(axis=1,inplace=True)
d

### export to hdf5-format for later plotting in other scripts
# We use hdf5 because it functions well with multiindex

d.to_hdf('dataproject/dataproject/Askers rod/data.h5', key='losses')






