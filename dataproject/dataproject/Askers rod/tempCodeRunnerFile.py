#data import

import pandas_datareader.data as web
import datetime
import pandas as pd
import numpy as np


# set timeframe we want to analyze:
start = datetime.datetime(2011, 1, 1)
end = datetime.datetime(2017, 1, 1)

# import data
f = web.DataReader(['TSLA','GOOG', 'MAERSK-A.CO'], 'yahoo', start, end)
# set index to date
f.index = pd.to_datetime(f.index)
# add column indicating year
f['year']=f.index.year
#rename column company to year
f.columns.rename(['Attributes', 'Company'],inplace  = True)
#show head
f.head()


#plotting:

from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource

source = ColumnDataSource(f)

source.data
p = figure()
p.line(x='Date', y='Close_GOOG', source=source)
show(p)