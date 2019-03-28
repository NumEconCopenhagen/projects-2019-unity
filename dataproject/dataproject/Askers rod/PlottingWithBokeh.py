import datetime
import pandas as pd
import numpy as np

#data import, from the HDF5-format:
d = pd.read_hdf('dataproject/dataproject/Askers rod/data.h5', key='losses')
d

#### Plotting
from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource

source = ColumnDataSource(d)

## Here we see how ColumnDataSource interprets the multiIndex
# it creates a bunch of arrays with names made of a combination of the multiindex
# for example Atribute='High' and Company='GOOG' becomes 'High_GOOG'
source.column_names


## lists can be made that only include a specific company og only includes af specific attribute
GOOG = [s for s in source.column_names if 'GOOG' in s]
GOOG



## graphing
output_file('runmean.html')

comp = "GOOG"

p = figure(x_axis_type='datetime',title=f'Graph with running means for {comp}', \
    tools="pan,box_zoom,reset,save", y_axis_label='Closing price', x_axis_label='Date')
p.line(x='Date', y=f'Close_{comp}', source=source, color = 'blue', legend= 'close')
p.line(x='Date', y=f'rm_200_{comp}', source=source, color='black', line_dash='4 4', legend = '200 days')
p.line(x='Date', y=f'rm_50_{comp}', source=source, color='red', line_dash='4 4',legend = '50 days')
p.legend.location = "top_left"

show(p)
