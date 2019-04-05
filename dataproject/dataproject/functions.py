### This is the file that defines all the functions we use in our final product
## imports we use for datareading and cleaning
import pandas_datareader.data as web
import datetime
import pandas as pd
import numpy as np


## The function that downloads data, it can take, multiple stocks and different years,
# depending on your need. You can also optionaly input a dictionairy,
# to automaticly change the Yahoo-symbol to the acutal names of the stocks and indices they represent. 

## check if dic og comp symbols exist
def download_data_with_runmean(stocks=['GOOG'], from_year = 2011 , to_year = 2017, stock_dict = {'GOOG':'Google'}):
    

    # We download an extra year back so we have running mean for the whole period 
    # (we have to go more than 200 days back because of holidays etc.)

    start_download = datetime.datetime((from_year-1), 1, 1)
    start_show = datetime.datetime((from_year), 1, 1)
    end = datetime.datetime(to_year, 12, 31)

    
    # Download data using pandas_datareader
    d = web.DataReader(stocks, 'yahoo', start_download, end)
    ## the DataFrame has the date as its index and in its column,
    # it actualy has an multiindex where both symbols and attribute has its own levels 

    #rename columns multiindex level Symbols to Stock, to make it more clear what it is
    d.columns.rename(['Attributes','Stock'],inplace  = True)

    # Reanme 'Adj_Close' to 'Adj_Close', this is because later when we reference this variable in bokeh hovertools
    # it doesn't respond well to strings with spaces
    d.rename(columns = {'Adj Close': 'Adj_Close'} ,inplace  = True)

    # Update strock-names from symbols. The dictionairy has to be passed as an argument,
    # so not all stocks will automaticly have it's real name, they'll just have theire symbols instead.
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


    # MultiIndexes have levels, right now Atributes is first, Stock is second
    # this line makes Stock, it changes the way you reference the index
    # right now d['name'] references an attribute (name has to be and attribute)
    # inputing a stock as name will cause an error
    # After this line d['name'] will reference and stock (and name has to be an stock of symbol of one)
    d = d.swaplevel(axis=1)
    #Sort DataFrame to make it look nicer. 
    d.sort_index(axis=1,inplace=True)

    # This line deletes the data for the years that weren't requested, 
    # but downloaded anyways in order to make the running means 
    d = d.loc[start_show:]

    return d


## bokeh and interactions imports 
from bokeh.io import output_notebook, push_notebook,show
from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource, HoverTool, NumeralTickFormatter
import ipywidgets as widgets
from ipywidgets import interact 

# This function plots the closing price of stocks in the DataFrame 'd' in an interactive way

def plot_close(d):

    ## Hovertool does so that when your mouse 'hovers' over the graph price and date is shown
    hover = HoverTool(tooltips=[('Price','@Adj_Close{0,0.0}'),('Volume','@Volume{0,0.}'),('Date','@Date{%F}')] ,formatters = {'Date':'datetime'})
    #  $y{0,0.0} is formatiing, "0," defines value y with comma (like 1000 is 1,000) and "0.0" is one decimal
    #  @x{%F} defines value x and use format option, fomatters then define that we use datetime format, 
    # since the x-value is the date.


    ## tools allows you to pan around, zoom in (you have to choose the boxzoom option in the graph then 
    # 'mark' area you wish to view) 
    ## reset to original position (there is no zoom in, besides this option),
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

    # Plot a line with the adjusted close plices, we reference our stocksource, that contains data for our choosen stock
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

        # push_notebook is af bokeh->jupyter-specific comand to tell python to update the bokeh plot in jupyter
        push_notebook()
    
    # show the plot, notebook_handle is again bokeh->jupyter-specific to show the plot in the notebook
    show(p,notebook_handle=True)

    # Make a dropdown-widget, so the user can choose different stocks, the layout option, makes the options box wider
    # the style options make the desciption box wide enough for the description string to be read fully
    drop_down = widgets.Dropdown(options=stock_list, layout = {'width':'50%'},\
        description='Choose a stock or index:',style = {'description_width': 'initial'})

    # Call the function 
    interact(update_name, stock = drop_down)
    



# This function plots multiple variables of a given stock in the DataFrame 'd' 
# in an interactive way so you can choose mulitple companies
# It is in a lot of ways similar to the plot_close() function, 
# so the explanation while be a bit more sparce and focus on added feature

def plot_close_mean(d):


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
    p.line(x='Date', y='rm_200', source=stocksource, color='indigo', line_dash='4 4', legend = '200 days',muted_alpha=0.2)
    p.line(x='Date', y='rm_50', source=stocksource, color='red', line_dash='4 4',legend = '50 days',muted_alpha=0.2)
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


#imports for making introduction plot
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
from matplotlib.transforms import Bbox
from matplotlib.path import Path
from matplotlib import style 
import matplotlib.dates
import matplotlib.animation as animation 
from mpl_finance import candlestick_ohlc
import mpl_finance 
from matplotlib.ticker import FuncFormatter


print(str(['TSLA'][0]))
def style_plot(stocksymbol = ['TSLA'], stockname = 'None' , from_year = 2015, to_year = 2017, save=False):
    
    if stockname == 'None':
        stockname = str(stocksymbol[0])
        if stocksymbol == ['TSLA']:
            stockname = 'Tesla'
        
            
    plt.style.use(["fivethirtyeight"])
    
    # Making our figure variables, and figure size
    fig = plt.figure(facecolor="white", figsize=(15, 10))
    fig2, ax2 = plt.subplots(figsize=(15, 10))
    
    # Setting global background settings for out graph. 
    plt.rcParams['axes.facecolor']='white'
    plt.rcParams['savefig.facecolor']='white'
    plt.rcParams["figure.edgecolor"]="white"
    
    # We are taking Teslas stock data from yahoo finance and setting the specific, 
    # datetime. Note that we can change "TSLA" to another company, and get the same, 
    # results. Although arrow and text won't change. 
    start = datetime.datetime(from_year, 1, 1)
    end = datetime.datetime(to_year, 12, 31)
    df = web.DataReader(stocksymbol, 'yahoo', start, end)

    # Specifying a format for date to match our data
    df.index = pd.to_datetime(df.index)
    df['year']= df.index.year
    df.rename(columns={'Symbols':'Company'},inplace=True)
    
    # Due to weird dates in matplotlib, we are changing them
    df["Date"]=matplotlib.dates.date2num(df.index.to_pydatetime())
    df
    # Making our two moving average, for 200- and 50 days
    # Specyfying min_periods to make it run smoother. 
    # (Can be usefull with bigger data-set)
    df["50ma"] = df["Adj Close"].rolling(window=50, min_periods=0).mean() 
    df["200ma"] = df["Adj Close"].rolling(window=200, min_periods=0).mean() 
    
    #Describing our two graphs.
    ax1 = plt.subplot2grid((7,1), (0,0), rowspan=5, colspan=1)
    ax2 = plt.subplot2grid((7,1), (5,0), rowspan=1, colspan=1)
    
    # Here we are making our candlestick-graph. 
    # Note: Candlestick are best used for shorter timeperiods, for better visuals. 
    ohlc_data = df[["Date",'Open','High','Low','Adj Close']].values
    candlestick_ohlc(ax1,ohlc_data,colorup='g',colordown='r',alpha=1)
    
  
    #Setting our dates for the two graphs. 
    ax1.xaxis.set_major_formatter(matplotlib.dates.DateFormatter(""))
    ax2.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y-%m"))
    
    # Here we are plotting our three graphs 
    ax1.plot(df.index, df["50ma"], color="teal", linestyle="--") 
    ax1.plot(df.index, df["200ma"], color="maroon", linestyle="--")
    ax2.plot(df.index, df["Volume"], color="black", fillstyle="none", linestyle="-")
    
    #Setting labels on our axis 
    plt.xlabel("Dates")
    plt.ylabel("Volume")
    ax1.set_ylabel("Price")
    
    # setting the upper left corner label 
    candle_patch_red = mpatches.Patch(color="red", label= f"{stockname} adj. closing stock price, Open < Close")
    candle_patch__green = mpatches.Patch(color="green", label=f"{stockname} adj. closing stock price, Open > Close")
    gold_patch = mpatches.Patch(color="maroon", label="200ma")
    silver_patch = mpatches.Patch(color="teal", label="50ma")
    volume_patch = mpatches.Patch(color="black", label="Volume")
    ax1.legend(handles=[ candle_patch__green, candle_patch_red, gold_patch, silver_patch,
                        volume_patch], loc="upper left", prop={'size': 15}, frameon=False)
    
    
    # Making a title for the graph. 
    ax1.set_title(f"Analyzing {stockname}'s stock price", size=30, color="Black", fontweight="bold")
    
    # Here we are using -annotate to making a arrow and text in the graph. 
    # We did set the cordinates for the text manually so it is only used for Tesla
    if stocksymbol == ['TSLA']:
        ax1.annotate('Death Cross', size=11, fontweight="bold",  xy=(datetime.date(2015, 11, 5), 200), 
                 xytext=(datetime.date(2015, 9, 15), 150),arrowprops=dict(facecolor='black'),)
        ax1.annotate('Golden Cross', size=11, fontweight="bold", xy=(datetime.date(2017, 2, 1), 200), 
                 xytext=(datetime.date(2016, 12, 25), 150), arrowprops=dict(facecolor='black'),)
    
    
    # These are only for the placed text in the graph. We did manually place them.    
    props = dict(boxstyle='Round', facecolor="wheat", alpha=1, edgecolor="red")
    props_source = dict(boxstyle='Round', facecolor="white")
    
    goldcross_text = "50day MA crosses \n above 200day MA \n signaling a change \n in momentum"
    ax1.text(0.76, 0.224, goldcross_text, transform=ax1.transAxes, fontsize=14,
            verticalalignment='center', bbox=props)
    deathcross_text = "50day MA crosses \n under 200day MA \n signaling a change \n in momentum"
    ax1.text(0.07, 0.11, deathcross_text, transform=ax1.transAxes, fontsize=14,
            verticalalignment='center')
    source_text = "Source: Yahoo-Finance"
    ax1.text(0.07, 0.11, deathcross_text, transform=ax1.transAxes, fontsize=14,
            verticalalignment='center', bbox=props)
    ax1.text(0.875, 0.0265, source_text, transform=ax1.transAxes, fontsize=10,
            verticalalignment='bottom', bbox=props_source)
    
    
    # Now we want to fill our ax2 graph.
    # We make the data to array
    volume = np.array(df["Volume"])
    # Then we need to make volume 1.Dimensional
    volume_1 = volume.ravel()
    # And we can then fill our ax2 figure
    ax2.fill_between(df.index, 0, volume_1, color="black")
    
    
    # USING CODE FROM SOURCE https://stackoverflow.com/questions/40566413/matplotlib-pyplot-auto-adjust-unit-of-y-axis
    # This code is automating the numbers on our axes. 
    # Our volume data is in millions, and it therefore sets a m behind 20. 
    def y_fmt(y, pos):
        decades = [1e9, 1e6, 1e3, 1e0, 1e-3, 1e-6, 1e-9 ]
        suffix  = ["G", "M", "k", "" , "m" , "u", "n"  ]
        if y == 0:
               return str(0)
        for i, d in enumerate(decades):
            if np.abs(y) >=d:
                val = y/float(d)
                signf = len(str(val).split(".")[1])
                if signf == 0:
                    return '{val:d} {suffix}'.format(val=int(val), suffix=suffix[i])
                else:
                    if signf == 1:
                        #print (val, signf)
                        if str(val).split(".")[1] == "0":
                           return '{val:d} {suffix}'.format(val=int(round(val)), suffix=suffix[i]) 
                    tx = "{"+"val:.{signf}f".format(signf = signf) +"} {suffix}"
                    return tx.format(val=val, suffix=suffix[i])
        #return y
    x_aa = np.linspace(0,349,num=350) 
    y = np.sinc((x_aa-66.)/10.3)**2*1.5e6+np.sinc((x_aa-164.)/8.7)**2*660000.+np.random.rand(len(x_aa))*76000.  
    #y = np.sinc((volume-66.)/10.3)**2*1.5e6+np.sinc((volume-164.)/8.7)**2*660000.+np.random.rand(len(volume))*76000.               
    ax1.yaxis.set_major_formatter(FuncFormatter(y_fmt))
    ax2.yaxis.set_major_formatter(FuncFormatter(y_fmt))
    
    
    # Here we are saving the graph directly to our desktop. With the title, 
    # Golden Cross. We are using a dpi=300, to set a better quality for the picture. 
    if save==True:
        plt.savefig("Golden Cross.png", facecolor=fig.get_facecolor(), edgecolor="b", dpi=300)
    
   
