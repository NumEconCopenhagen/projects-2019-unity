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

# a method to reference index in ft
ft.xs('GOOG',level='Company').head()

#method to reference index in f
#drop_levels=True means all column-names are saved ('Company')
f.xs('GOOG',level='Company',axis=1, drop_level=False).head()

# reference using loc:
ft.loc[(slice(None),slice('GOOG')),:].T.head()

#normalize to first price and plot:
close_norm = f.transform(lambda x: x/x[0])['Close']
ax = close_norm.plot()

# make running mean for 200 days:
ax = plt.subplot(111)
f.rolling(200).mean()['Close']['GOOG'].plot(ax=ax)
f['Close']['GOOG'][199:].plot(ax=ax)

# make series of runmean(200) and clossing price:
runmean = f.rolling(200).mean()['Close']['GOOG'][199:].rename('runmean')
close = f['Close']['GOOG'][199:].rename('close')

#put series in on DataFrame:
googlegraph = pd.DataFrame([runmean,close])

# make array that is True when the two graphs cross
cross = np.isclose(googlegraph.loc['runmean'], googlegraph.loc['close'],atol=1)
#turn in into a series with data (google.columns) as it's index
cross_series = pd.Series(data=cross, name='cross', index=googlegraph.columns)

#This can then be appended to the googlegraph
googlegraph = googlegraph.append(cross_series)

#show 

# plot
googlegraph.T.plot()
