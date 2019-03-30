### This is the file that defines all the functions we use in our final product
## imports we use
import pandas_datareader.data as web
import datetime
import pandas as pd
import numpy as np


## The function that downloads data, it can take, multiple companys and different years depending on your need
def download_data_with_runmean(companys=['GOOG'], from_year = 2011 , to_year = 2017):
    
    # We download an extra year back so we have running mean for the whole period 
    # (we have to go more than 200 days back because of holidays etc.)

    start_download = datetime.datetime((from_year-1), 1, 1)
    start_show = datetime.datetime((from_year), 1, 1)
    end = datetime.datetime(to_year, 12, 31)

    
    # Download data
    d = web.DataReader(companys, 'yahoo', start_download, end)

    #rename columns Symbols to Company 
    d.columns.rename(['Attributes','Company'],inplace  = True)
    d.interpolate(inplace = True)

    # make 200 and 50 days running mean :
    rm_200 = d.rolling(200).mean()['Close']
    rm_50 = d.rolling(50).mean()['Close']

    ## these two new DataFrames are now concated to one with a multiIndex
    rm = pd.concat((rm_200,rm_50), keys=['rm_200', 'rm_50'], names=['Attributes', 'Company'], axis=1)

    # Which can then be merged to the main DataFrame, 
    # # using only join since both have same multiindex (in column) and index(Date)
    d = d.join(rm)


    # MultiIndexes have levels, right now Atributes is first, Symbols(company)is second
    # this line makes Symbos first 
    d = d.swaplevel(axis=1)
    #Sort DataFrame
    d.sort_index(axis=1,inplace=True)


    d = d.loc[start_show:]

    return d

from bokeh.io import output_notebook, push_notebook,show
from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource, HoverTool
from ipywidgets import interact


# This function plots the closing price of companys in the DataFrame 'd' in an interactive way

def plot_close(d):

    ## Hovertool does so that when your mouse 'hovers' over the graph price and date is shown
    hover = HoverTool(tooltips=[('Price','@Close{0.0}'),('Volume','@Volume'),('Date','@Date{%F}')] ,formatters = {'Date':'datetime'})
    #  $y{0.0} defines value y with one decimal
    #  @x{%F} defines value x and use format option, fomatters then define that we use datetime format, 
    # since the x-value is the date.


    ## tools allows you to pan around, zoom in (you have to choose the boxzoom option in the graph then 
    # 'mark' area you wish to view) 
    ## reset to original position (there is no zoom in, besides this option),
    # and save a the graph as a picture
    tools="pan,box_zoom,reset,save"

    company_list = list(d.columns.levels[0])
    company = list(d.columns.levels[0])[0]
    companysource = ColumnDataSource(d[company])
    p = figure(x_axis_type='datetime',title=f'Closing price of {company}', \
        tools=[hover,tools], y_axis_label='Closing price', x_axis_label='Date')

    p.line(x='Date', y='Close', source=companysource, color = 'blue', legend= 'Closing price')
    p.legend.location = "top_left"
    p.ygrid.minor_grid_line_color = 'navy'
    p.ygrid.minor_grid_line_alpha = 0.1

    def update_name(company):
        p.title.text = f'Closing price of {company}'
        companysource.data = ColumnDataSource(d[str(company)]).data
        push_notebook()
    
    show(p,notebook_handle=True)


    interact(update_name,company=company_list)




def plot_close_mean(d):


    ## Hovertool does so that when your mouse 'hovers' over the graph price and date is shown
    hover = HoverTool(tooltips=[('Closing price','@Close{0.0}'),('50 days','@rm_50{0.0}'),\
        ('200 days','@rm_200{0.0}'),('Volume','@Volume'),('Date','@Date{%F}')] , formatters = {'Date':'datetime'})
    #  $y{0.0} defines value y with one decimal
    #  @x{%F} defines value x and use format option, fomatters then define that we use datetime format, 
    # since the x-value is the date.


    ## tools allows you to pan around, zoom in (you have to choose the boxzoom option in the graph then 
    # 'mark' area you wish to view) 
    ## reset to original position (there is no zoom in, besides this option),
    # and save a the graph as a picture
    tools="pan,box_zoom,reset,save"


    company_list = list(d.columns.levels[0])
    company = list(d.columns.levels[0])[0]
    companysource = ColumnDataSource(d[company])
    p = figure(x_axis_type='datetime',title=f'Closing price and running mean of {company}', \
        tools=[hover,tools], y_axis_label='Closing price', x_axis_label='Date')

    p.line(x='Date', y='Close', source=companysource, color = 'blue', legend= 'Closing price')
    p.line(x='Date', y='rm_200', source=companysource, color='black', line_dash='4 4', legend = '200 days')
    p.line(x='Date', y='rm_50', source=companysource, color='red', line_dash='4 4',legend = '50 days')
    p.legend.location = "top_left"
    p.ygrid.minor_grid_line_color = 'navy'
    p.ygrid.minor_grid_line_alpha = 0.1

    def update_name(company):
        p.title.text = f'Closing price and running mean of {company}'
        companysource.data = ColumnDataSource(d[str(company)]).data
        push_notebook()
    
    show(p,notebook_handle=True)


    interact(update_name,company=company_list)