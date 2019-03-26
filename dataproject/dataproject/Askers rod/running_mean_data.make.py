import pandas_datareader.data as web
import datetime
import pandas as pd
import numpy as np


# set timeframe we want to analyze:
start = datetime.datetime(2011, 1, 1)
end = datetime.datetime(2017, 1, 1)

### Data import ####

# import data
d = web.DataReader(['TSLA', 'GOOG', 'MAERSK-A.CO'], 'yahoo', start, end)
# set index to date
d.index = pd.to_datetime(d.index)
# add column indicating year
d['year']=d.index.year
#rename column company to year
d.columns.rename(['Attributes', 'Company'],inplace  = True)
#show head
d.head()



### Data cleaning ###
'''
Because the Danish and American stockexchange are sometimes open on different days, 
data is missing for some days when one is open and the other is not
for these days the NaN (missing value) will be replaced with the value of the previous day
'''

d.reindex(df.index[::-1]).ffill()



### Creating aditional columns ###

# make 200 days running mean :
rm_200 = d.rolling(200).mean()['Close']
rm_50 = d.rolling(50).mean()['Close']
#checking for nonacepted means (x-coordinates should go no higher than 200 and 50)
np.where(pd.isnull(rm_200))
np.where(pd.isnull(rm_50))

