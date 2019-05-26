### This is the file that defines all the functions we use in our final product
## imports we use for datareading and cleaning
import pandas_datareader.data as web
import datetime
import pandas as pd
import numpy as np


## check if dic og comp symbols exist
def download_data(stocks=['GOOG'], from_year = 2011 , to_year = 2017, 
            stock_dict = {'GOOG':'Google'}, weights=[], printit=True):
    '''
    Downloads stock- data from the yahoo database, 
    calculates the 200- and 50-days running means, the total run on the stock from start til end,
    and also the return from investing in the stock using the golden/death-cross-strategy in the period.
    During the datacleaning process, this funtion also prints the data between calculations.
    You can also optionaly input a dictionairy, to automaticly change the ticker-symbol,
    to the acutal names of the stocks and indixes they represent. 

    Args:
        stocks (list of strings)    : A list contaigning the ticker symbols for the stocks to include in the dataset
        from_year,to_year (int)     : Period to include in the data set, the year before is also downloaded in order to
                                        calculate all runnning means, but deleted before being returned 
        stock_dict (dict) (optional): Containing ticker-symbol and name of stock.

    Returns:
        d (pd.DataFrame)            : The stock data, utilizes multiindex in the column, to include mulitple information on each stock
    
    
    '''

    # We download an extra year back so we have running mean for the whole period 
    # (we have to go more than 200 days back because of holidays etc.)

    start_download = datetime.datetime((from_year-1), 1, 1)
    start_show = datetime.datetime((from_year), 1, 1)
    end = datetime.datetime(to_year, 12, 31)

    
    # Download data using pandas_datareader
    d = web.DataReader(stocks, 'yahoo', start_download, end)
    ## the DataFrame has the date as its index and in its column,
    # is actualy an multiindex where both symbols and attribute has its own levels

    #rename columns multiindex level Symbols to Stock, to make it more clear what it is
    d.columns.rename(['Attributes','Stock'],inplace  = True)

    if printit:
        print(f'This is the head of our initially downloaded dataset for {stocks[0]}:')
        print(d.xs(stocks[0],level='Stock',axis=1).head())
        d.xs(stocks[0],level='Stock',axis=1).head()

    # Reanme 'Adj_Close' to 'Adj_Close', this is because later when we reference this variable in bokeh hovertools
    # it doesn't respond well to strings with spaces
    d.rename(columns = {'Adj Close': 'Adj_Close'} ,inplace  = True)

    # Update strock-names from symbols. The dictionairy has to be passed as an argument,
    # so not all stocks will automaticly have it's real name, 
    # if it's not defined in the dict, they'll just have their symbols instead.
    d.rename(columns = stock_dict, inplace = True)

    # Replace missing values with mean of the values from the day before and the day after
    # The reason for missing values is, that if stocks from different countries is chosen
    # Some days will have missing values because of holidays in one country that is a workday in other.
    # The limit 3 is arbitrairily chosen, it just means that if 3 or more days in a row are missing it will still be a NaN
    # This is to avoid interpolation over closed stocks.
    d.interpolate(limit=3, inplace = True)
    # We need this in order to make the running means, since the functions requires there to be no NaNs
    #     
    # Make 200 and 50 days running mean:
    rm_200 = d.rolling(200).mean()['Adj_Close']
    rm_50 = d.rolling(50).mean()['Adj_Close']

    # The way we do it we acutualy make running means for all the variables both only safe the Adj_close

    ## these two new DataFrames are now concated to one with a multiIndex that corresponds to d
    rm = pd.concat((rm_200,rm_50), keys=['rm_200', 'rm_50'], names=['Attributes', 'Stock'], axis=1)

    # Which can then be merged to the main DataFrame, d
    # using only join since both have same multiindex (in column) and index(Date)
    d = d.join(rm)
   

    # This line deletes the data for the years that weren't requested, 
    # but downloaded anyways in order to make the running means 
    d = d.loc[start_show:]

    if printit:
        print('\n \n')
        print('Now we have added running means and deleted the earliest observations, that was used to create them:')
        print(d.xs(stock_dict[stocks[0]],level='Stock',axis=1).head())

    # Here we caluate the collected return on investing in the stock from 
    # the starting date till the end, and also the return on investing in 
    # accordance with the stock theory:
     

    #overall return:
    returns_cum = ((d.pct_change()["Adj_Close"]+1).cumprod()-1)*100
    returns = d["Adj_Close"].pct_change()

    buy_signal =  d['rm_50']-d['rm_200']
    where = np.where(buy_signal>0,1,0)
    strategy = where*returns
    #strategy return:
    strategy_cum = ((strategy+1).cumprod()-1)*100
    
    # Make a dataframe of data and merge the sets:

    returns_data = pd.concat((returns_cum,strategy_cum), keys=['Returns_cum', 'Strategy_cum'], names=['Attributes', 'Stock'], axis=1)
    d = d.join(returns_data)


    # MultiIndexes have levels, right now Atributes is first, Stock is second
    # this line makes Stock, it changes the way you reference the index
    # right now d['name'] references an attribute (name has to be and attribute)
    # inputing a stock as name will cause an error
    # After this line d['name'] will reference a stock (and name has to be an stock of symbol of one)
    d = d.swaplevel(axis=1)

    #Sort DataFrame to make it look nicer. 
    d.sort_index(axis=1,inplace=True)


    if len(weights)==0:
        return d
    else:
        a = strategy_cum*weights
        portefolio_strategy_return = a.sum(axis=1)
        b=returns_cum*weights
        portefolio_return = b.sum(axis=1)
        return d, portefolio_strategy_return, portefolio_return


## bokeh and interactions imports 
from bokeh.io import output_notebook, push_notebook,show
from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource, HoverTool, NumeralTickFormatter
import ipywidgets as widgets
from ipywidgets import interact 



def plot_close(d):
    '''
    This function plots the closing price of stocks in the DataFrame 'd' interactively,
    using bokeh and ipywidgets.

    Args:
        d (pd.DataFrame)   : Containing adjusted closing price and volume for all companys that is to be plotted. 

    '''
    ## Hovertool does so that when your mouse 'hovers' over the graph price and date is shown
    hover = HoverTool(tooltips=[('Price','@Adj_Close{0,0.0}'),('Volume','@Volume{0,0.}'),('Date','@Date{%F}')] ,formatters = {'Date':'datetime'})
    #  $y{0,0.0} is formatiing, "0," defines value y with comma (like 1000 is 1,000) and "0.0" is one decimal
    #  @x{%F} defines value x and use format option, fomatters then define that we use datetime format, 
    # since the x-value is the date.


    ## tools allows you to pan around, zoom in (you have to choose the boxzoom option in the graph then 
    # 'mark' area you wish to view) 
    ## reset to original position (there is no zoom out, besides this option),
    # and save a the graph as a picture
    tools="pan,box_zoom,reset,save"

    # Make a list of companys we have data for
    stock_list = list(d.columns.levels[0])
    # Choose our inital stock
    stock = list(d.columns.levels[0])[0]

    # This line creates a source for bokeh plotting with our chosen stock 
    stocksource = ColumnDataSource(d[stock])
    
    # We initialize our figure with titles, labels and specify that we want to use our tools mentioned above
    p = figure(x_axis_type='datetime',title=f'Closing price of {stock}', \
        tools=[hover,tools], y_axis_label='Closing price', x_axis_label='Date')

    # Plot a line with the adjusted close prices, we reference our stocksource, that contains data for our choosen stock
    p.line(x='Date', y='Adj_Close', source=stocksource, color = 'blue', legend= 'Closing price')

    # We locate the legend in the top left, stocks mostly go from bottom left corner to top right corner
    # so we figured this was a good location
    p.legend.location = "top_left"

    # visuals 
    p.ygrid.minor_grid_line_color = 'navy'
    p.ygrid.minor_grid_line_alpha = 0.1
    # formatting y-axis:
    p.yaxis.formatter=NumeralTickFormatter(format='0,0')

    # Defines a function that updates the plot when a different stock is chosen
    def update_name(stock):
        p.title.text = f'Closing price of {stock}'
        # Updates the data
        stocksource.data = ColumnDataSource(d[str(stock)]).data
        # I can't for the life of my explain why .data needs to there on both side of the equal sign
        # both they do

        # push_notebook is a bokeh->jupyter-specific comand to tell python to update the bokeh plot in jupyter
        push_notebook()
    

    # show the plot, notebook_handle is bokeh->jupyter-specific to show the plot in the notebook
    show(p,notebook_handle=True)

    # Make a dropdown-widget, so the user can choose different stocks, the layout option, makes the options box wider
    # the style options make the desciption box wide enough for the description string to be read fully
    drop_down = widgets.Dropdown(options=stock_list, layout = {'width':'50%'},\
        description='Choose a stock or index:',style = {'description_width': 'initial'})

    
    interact(update_name, stock = drop_down)
    




def plot_close_mean(d):
    '''
    This function plots the closing price of stocks and the running means,
    in the DataFrame 'd' interactively, using bokeh and ipywidgets.

    Args:
        d (pd.DataFrame)   : Containing adjusted closing price,running means and volume for all companys that is to be plotted. 

    '''

    ## Hovertool does so that when your mouse 'hovers' over the graph price and date is shown
    hover = HoverTool(tooltips=[('Closing price','@Adj_Close{0,0.0}'),('50 days','@rm_50{0,0.0}'),\
        ('200 days','@rm_200{0,0.0}'),('Volume','@Volume{0,0.}'),('Date','@Date{%F}')] , formatters = {'Date':'datetime'})
    #  $y{0.0} defines value y with one decimal
    #  @x{%F} defines value x and use format option, fomatters then define that we use datetime format, 
    # since the x-value is the date.

    ## tools allows you to pan around, zoom in (you have to choose the boxzoom option in the graph then 
    # 'mark' area you wish to view) 
    ## reset to original position (there is no zoom in, besides this option),
    # and save a the graph as a picture
    tools="pan,box_zoom,reset,save"


    stock_list = list(d.columns.levels[0])
    stock= list(d.columns.levels[0])[0]
    stocksource = ColumnDataSource(d[stock])
    p = figure(x_axis_type='datetime',title=f'Closing price and running mean of {stock}', \
        tools=[hover,tools], y_axis_label='Closing price', x_axis_label='Date')


    # The muted_alpha option is for an exstra interaction, if you click on a variable name in the legend of the plot
    # The varible is 'muted' by making the plot more transparent, this option defines how much
    p.line(x='Date', y='Adj_Close', source=stocksource, color = 'blue', legend= 'Closing price',muted_alpha=0.3,line_width=0.5)
    # plotting mulitple lines is possible in bokeh whuhu
    # the line_dash option makes the line plotted like - - - - instead of a straight line. 
    p.line(x='Date', y='rm_200', source=stocksource, color='red', line_dash='4 4', legend = '200 days',muted_alpha=0.2)
    p.line(x='Date', y='rm_50', source=stocksource, color='indigo', line_dash='4 4',legend = '50 days',muted_alpha=0.2)
    p.legend.location = "top_left"
    p.ygrid.minor_grid_line_color = 'navy'
    p.ygrid.minor_grid_line_alpha = 0.1
    p.yaxis.formatter=NumeralTickFormatter(format='0,0')
 
    # This activates the muting option specficed while ploting 
    p.legend.click_policy="mute"

    # The rest is almost exactly the same as plot_clos()
    def update_name(stock):
        p.title.text = f'Closing price and running mean of {stock}'
        stocksource.data = ColumnDataSource(d[str(stock)]).data
        push_notebook()
    
    show(p,notebook_handle=True)

    
    drop_down = widgets.Dropdown(options=stock_list, layout = {'width':'50%'},\
        description='Choose a stock or index:',style = {'description_width': 'initial'})
    interact(update_name, stock = drop_down)


def plot_returns(d):
    '''
    This function plots the returns and returns of golden/death-strategy,
    in the DataFrame 'd' interactively, using bokeh and ipywidgets.

    Args:
        d (pd.DataFrame)   : Containing returns on stock, returns on strategy and volume for all companys that is to be plotted. 

    '''


    ## Hovertool does so that when your mouse 'hovers' over the graph price and date is shown
    hover = HoverTool(tooltips=[('Return on stock','@Returns_cum{0,0.00}%'),('Return on strategy','@Strategy_cum{0,0.00}%'),\
        ('Volume','@Volume{0,0.}'),('Date','@Date{%F}')] , formatters = {'Date':'datetime'})
    #  $y{0.0} defines value y with one decimal
    #  @x{%F} defines value x and use format option, fomatters then define that we use datetime format, 
    # since the x-value is the date.

    ## tools allows you to pan around, zoom in (you have to choose the boxzoom option in the graph then 
    # 'mark' area you wish to view) 
    ## reset to original position (there is no zoom in, besides this option),
    # and save a the graph as a picture
    tools="pan,box_zoom,reset,save"


    stock_list = list(d.columns.levels[0])
    stock= list(d.columns.levels[0])[0]
    stocksource = ColumnDataSource(d[stock])
    p = figure(x_axis_type='datetime',title=f'Testing our theory on {stock}', \
        tools=[hover,tools], y_axis_label='Cumulative rate of return, %', x_axis_label='Date')


    # The muted_alpha option is for an exstra interaction, if you click on a variable name in the legend of the plot
    # The varible is 'muted' by making the plot more transparent, this option defines how much

    
    p.line(x='Date', y='Returns_cum', source=stocksource, color='indigo', legend = 'Market',muted_alpha=0.2)
    p.line(x='Date', y='Strategy_cum', source=stocksource, color='red',legend = 'Strategy',muted_alpha=0.2)
    p.legend.location = "top_left"
    p.ygrid.minor_grid_line_color = 'navy'
    p.ygrid.minor_grid_line_alpha = 0.1
    p.yaxis.formatter=NumeralTickFormatter(format='0,0')

    # This activates the muting option specficed while ploting 
    p.legend.click_policy="mute"

    # The rest is almost exactly the same as plot_clos()
    def update_name(stock):
        p.title.text = f'Testing our theory on {stock}'
        stocksource.data = ColumnDataSource(d[str(stock)]).data
        push_notebook()
    
    show(p,notebook_handle=True)

    
    drop_down = widgets.Dropdown(options=stock_list, layout = {'width':'50%'},\
        description='Choose a stock or index:',style = {'description_width': 'initial'})
    interact(update_name, stock = drop_down)



def plotting(x,y_names,  x_array, y_arrays,y_name, title='Figure',
                colors= ['red','blue','green','purple','yellow'],
                legendlocation="top_center",tools="pan,wheel_zoom,box_zoom,reset,save",
                width=450, height=450, y_unit='',line_width=1.5): 
    
    '''
    Makes lineplots of the arrays using bokeh

    Args:
            x(string)       : Name of x-axis
            y_names (list)  : Containing strings with the names of the lineplots
            x_array(array)  : Data for x-variable
            y_arrays(list)  : Containing arrays with data for the y-variable of all plots
            y_name(string)  : Name of y_axis
            title(string)   : Figure title
            colors (list)   : list of colors of the lines, code for Hex colors is also accepted
            legendlocation(string): location of legend, written as "horizontalspace_verticalspace"
                                    Horizontalspace can be bottom  or center
                                    Verticalspace can be left, middle or right
            tools(string)       : Bokeh interactive tools, for the plot
            width,height (ints) : Determines the size of the figure
            y_unit(string)      : Unit of the y-axis data like % or $
            line_width (float)  : Width of the lines that are plotted
    
    Returns:
            p (bokeh.plotting.figure.Figure): The figure, which has to be called 
            in the bokeh.plotting comand, show(), to be viewed
    '''
    
    # Bokeh needs a name for the data that neither has spaces nor numbers
    # because we want the option to do this we define abitrairy calls via the alphabeth. 
    calls = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 
     'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    y_calls = []

    for i in range(len(y_names)):
        y_calls.append(calls[i])
    
    #Hover
    tooltips=[]
    
    for yn,yc in zip(y_names,y_calls):
        if yn !='' :
            text = '@'+f'{yc}'+'{0.00}'+y_unit
            tooltips.append((f'{yn}',text))
    
    hover = HoverTool(tooltips=tooltips)

    #Data
    data = {'x': x_array}
    for yc, y_array in zip(y_calls,y_arrays):
        data[f'{yc}']=y_array
        
    source = ColumnDataSource(data)
    
    #Figure and plotting lines
    p = figure(plot_width=width, plot_height=height, title=f'{title}', 
        tools=[hover,tools], x_axis_label=f'{x}', y_axis_label=f'{y_name}',
        x_axis_type="datetime")

    for i,(yc,yn) in enumerate(zip(y_calls,y_names)):
        p.line(x='x', y=f'{yc}', source=source, legend=f'{yn}', 
        color = colors[i],line_width=line_width)
    

    p.legend.location = legendlocation
    
    return p 