import datetime
import pandas as pd
import numpy as np

#data import, from the HDF5-format:
d = pd.read_hdf('dataproject/dataproject/Askers rod/data.h5', key='losses')
d

# for later reference we make lists of our options
company_list = list(d.columns.levels[0])
attribute_list = list(d.columns.levels[1])


#### Plotting

#imports for plot
from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource, Select
from bokeh.io import curdoc
# imports for interaction
from bokeh.layouts import widgetbox, row
from bokeh.models.widgets import Select

source = ColumnDataSource(d)

## Here we see how ColumnDataSource interprets the multiIndex
# it creates a bunch of arrays with names made of a combination of the multiindex
# for example Atribute='High' and Company='GOOG' becomes 'High_GOOG'
source.column_names


## lists can be made that only include a specific company og only includes af specific attribute
GOOG = [s for s in source.column_names if 'GOOG' in s]
GOOG

d = d.swaplevel(axis=1)

source.data
np.array(d['Close']['GOOG'])


close = ColumnDataSource(data= \
    {'x' : np.array(d.index), 'y' : np.array(d['Close']['GOOG'])})

p = figure(x_axis_type='datetime',title=f'Graph with close', \
    tools="pan,box_zoom,reset,save", y_axis_label='Closing price', x_axis_label='Date')
p.line(x='x', y='y', source=close, color = 'blue', legend='close')

show(p)
