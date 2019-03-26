import pandas_datareader.data as web
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# set timeframe we want to analyze:
start = datetime.datetime(2011, 1, 1)
end = datetime.datetime(2017, 1, 1)

# import data
f = web.DataReader(['TSLA','GOOG'], 'yahoo', start, end)
# set index to date
f.index = pd.to_datetime(f.index)
# add column indicating year
f['year']=f.index.year
#rename column company to year
f.columns.rename(['Attributes', 'Company'],inplace  = True)
#show head
f.head()
#save transformed verison, this will be usefull for reference later on
ft = f.T
# show parameters of multiindex:
ft.index
#this is the same as:
f.columns


# plot clossing price for entire period
ax = f['Close'].plot()