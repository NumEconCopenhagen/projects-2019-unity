import datetime
import pandas as pd
import numpy as np

#data import, from the HDF5-format:
d = pd.read_hdf('dataproject/dataproject/Askers rod/data.h5', key='losses')
d

#### Plotting
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource

source = ColumnDataSource(d)
source.data

p = figure(x_axis_type="datetime")
p.line(x='Date', y='Close_GOOG', source=source)
p.line(x='Date', y='rm_200_GOOG', source=source, color='black')
p.line(x='Date', y='rm_50_GOOG', source=source, color='red')
show(p)
