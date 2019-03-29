import datetime
import pandas as pd
import numpy as np

#data import, from the HDF5-format:
d = pd.read_hdf('dataproject/dataproject/Askers rod/data.h5', key='losses')
d

# for later reference we make lists of our options
company_list = list(d.columns.levels[1])
attribute_list = list(d.columns.levels[0])


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

source.data
np.array(d['Close']['GOOG'])


close = ColumnDataSource(data= \
    {'x' : np.array(d.index), 'y' : np.array(d['Close']['GOOG'])})

p = figure(x_axis_type='datetime',title=f'Graph with close', \
    tools="pan,box_zoom,reset,save", y_axis_label='Closing price', x_axis_label='Date')
p.line(x='x', y='y', source=close, color = 'blue', legend='close')

show(p)

## graphing
output_file('runmean.html')

comp = company_list[0]

p = figure(x_axis_type='datetime',title=f'Graph with running means for {comp}', \
    tools="pan,box_zoom,reset,save", y_axis_label='Closing price', x_axis_label='Date')
p.line(x='Date', y=f'Close_{comp}', source=source, color = 'blue', legend= 'close')
p.line(x='Date', y=f'rm_200_{comp}', source=source, color='black', line_dash='4 4', legend = '200 days')
p.line(x='Date', y=f'rm_50_{comp}', source=source, color='red', line_dash='4 4',legend = '50 days')
p.legend.location = "top_left"


## interactive
# For specific companys, could make options list dependent on source.column_names
select = Select(title='Company:', value=company_list[0], options=company_list)



show(p)
