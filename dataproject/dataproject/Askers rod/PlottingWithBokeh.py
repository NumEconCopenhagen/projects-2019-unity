#data import

import pandas_datareader.data as web
import datetime
import pandas as pd
import numpy as np


# set timeframe we want to analyze:
start = datetime.datetime(2011, 1, 1)
end = datetime.datetime(2017, 1, 1)

# import data
d = web.DataReader(['TSLA','GOOG', 'MAERSK-A.CO'], 'yahoo', start, end)
# set index to date
d.index = pd.to_datetime(d.index)
# add column indicating year
d['year']=d.index.year
#rename column company to year
d.columns.rename(['Attributes', 'Company'],inplace  = True)
#show head
d.head()


#plotting:

from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource

source = ColumnDataSource(d)

source.data
p = figure()
p.line(x='Date', y='Close_GOOG', source=source)
show(p)