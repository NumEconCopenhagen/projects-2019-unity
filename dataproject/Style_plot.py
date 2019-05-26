
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import style 
import matplotlib.dates
from mpl_finance import candlestick_ohlc 
import datetime as dt
import pandas_datareader.data as web
import pandas as pd
import matplotlib.patches as mpatches



def figure_1(from_year = 2015, to_year = 2018):

    
##############################################################################
    # 5 Parts in our style_project
    # 1: Getting data                                                             
    # 2: Plotting introduction to our strategy                                    
    # 3: Calculating the returns                                                    
    # 4: How did our strategy do?                                                  
    # 5: Conclusion                                                               
##############################################################################
    
    
    '''
    First we are setting global settings, that will effect all the figures,
    we are making. We want grid, and the same figure size on all of them. 
    We will use this later. 
    '''
    plt.rcParams["axes.grid"]=True
    plt.rcParams["figure.figsize"]= 15, 10
    
    
    
    
########## Part 1: Getting data ##############################################
    
    
    '''
    We are taking Teslas stock data from yahoo finance and setting the specific, 
    datetime. Note that we can change "TSLA" to another company, and get the same, 
    results. (Although the arrows and text won't change). 
    '''
    start = dt.datetime(from_year, 1, 1)
    end = dt.datetime(to_year, 1, 1)
    df = web.DataReader(['TSLA'], 'yahoo', start, end)
    df["Date"]=matplotlib.dates.date2num(df.index.to_pydatetime())
    
    ''' Making our two moving averages.''' 
    df["50ma"] = df["Adj Close"].rolling(window=50, min_periods=0).mean() 
    df["200ma"] = df["Adj Close"].rolling(window=200, min_periods=0).mean() 
    
    '''Describing our two graphs.'''
    ax1 = plt.subplot2grid((7,1), (0,0), rowspan=5, colspan=1)
    plt.ylabel("Stock price")
    ax2 = plt.subplot2grid((7,1), (5,0), rowspan=1, colspan=1)
    
    '''Here we are making the candlestick graph.''' 
    ohlc_data = df[["Date",'Open','High','Low','Adj Close']].values
    candlestick_ohlc(ax1,ohlc_data,colorup='g',colordown='r',alpha=1)
    
    
    
    
########## Part 2: Plotting introduction to our strategy #####################
    
    
    
    
    ''' Here we are plotting 50ma, 200ma and volume.''' 
    ax1.plot(df.index, df["50ma"], color="teal", linestyle="--") 
    ax1.plot(df.index, df["200ma"], color="maroon", linestyle="--")
    ax2.plot(df.index, df["Volume"], color="black")
    
    '''Now we want to fill our ax2 graph (The volume). We make the data to an array'''
    volume = np.array(df["Volume"])
    '''Then we need to make volume 1.Dimensional'''
    volume_1 = volume.ravel()
    ''' We then fill our ax2 graph. '''
    ax2.fill_between(df.index, 0, volume_1, color="black")
    
    '''Here we are adjusting the dates on the figure'''
    ax1.xaxis.set_major_formatter(matplotlib.dates.DateFormatter(""))
    ax2.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y-%m"))
    
    '''Setting title and labels'''
    ax1.set_title("Analyzing Tesla's stock price", size=30, color="Black", fontweight="bold")
    
    plt.xlabel("Date")
    plt.ylabel("Volume")
    
    '''Setting the upper left corner label''' 
    candle_patch_red = mpatches.Patch(color="red", label= "Tesla adj. closing stock price, Open < Close")
    candle_patch__green = mpatches.Patch(color="green", label="Tesla adj. closing stock price, Open > Close")
    gold_patch = mpatches.Patch(color="maroon", label="200ma")
    silver_patch = mpatches.Patch(color="teal", label="50ma")
    volume_patch = mpatches.Patch(color="black", label="Volume")
    ax1.legend(handles=[ candle_patch__green, candle_patch_red,
                        gold_patch, silver_patch,
                        volume_patch], 
    loc="upper left", prop={'size': 15}, frameon=False)
    '''
    Here we are using -annotate to make an arrow and text in the graph. 
    We did set the cordinates for the text manually.
    '''
    ax1.annotate('Death Cross', size=11, fontweight="bold",  xy=(dt.date(2015, 11, 15), 200), 
                 xytext=(dt.date(2015, 9, 15), 150),
                arrowprops=dict(facecolor='black'),
                )
    ax1.annotate('Golden Cross', size=11, fontweight="bold", xy=(dt.date(2017, 2, 1), 200), 
                 xytext=(dt.date(2016, 12, 25), 150),
                arrowprops=dict(facecolor='black'),
                )
    '''These are only for the placed text in the graph. We did manually place them.'''    
    box_style = dict(boxstyle='Round', facecolor="wheat", alpha=1, edgecolor="red")
    source_style = dict(boxstyle='Round', facecolor="white")
    
    goldcross_text = "50day MA crosses \n above 200day MA \n signaling a change \n in momentum"
    deathcross_text = "50day MA crosses \n under 200day MA \n signaling a change \n in momentum"
    
    
    ax1.text(0.76, 0.224, goldcross_text, transform=ax1.transAxes, fontsize=14,
            verticalalignment='center', bbox=box_style)
    ax1.text(0.07, 0.11, deathcross_text, transform=ax1.transAxes, fontsize=14,
            verticalalignment='center')
    source_text = "Source: Yahoo-Finance"
    
    
    ax1.text(0.07, 0.11, deathcross_text, transform=ax1.transAxes, fontsize=14,
            verticalalignment='center', bbox=box_style)
    ax1.text(0.85, 0.03, source_text, transform=ax1.transAxes, fontsize=10,
            verticalalignment='bottom', bbox=source_style)
    
    
    ######### Using code from source: ############################################# 
    #https://stackoverflow.com/questions/40566413/matplotlib-pyplot-auto-adjust-unit-of-y-axi 
    def y_fmt(y, pos):
        decades = [1e9, 1e6, 1e3, 1e0, 1e-3, 1e-6, 1e-9 ]
        suffix  = ["G", "M$", "k", "$" , "m" , "u", "n"  ]
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
                        if str(val).split(".")[1] == "0":
                           return '{val:d} {suffix}'.format(val=int(round(val)), suffix=suffix[i]) 
                    tx = "{"+"val:.{signf}f".format(signf = signf) +"} {suffix}"
                    return tx.format(val=val, suffix=suffix[i])
    x_aa = np.linspace(0,349,num=350) 
    y = np.sinc((x_aa-66.)/10.3)**2*1.5e6+np.sinc((x_aa-164.)/8.7)**2*660000.+np.random.rand(len(x_aa))*76000.  
    from matplotlib.ticker import FuncFormatter
    ax1.yaxis.set_major_formatter(FuncFormatter(y_fmt))
    ax2.yaxis.set_major_formatter(FuncFormatter(y_fmt))
    ###############################################################################
    return df

########## Part 3: Calculating the returns ####################################

def figure_2(df):
    
    '''
    First we make a new dataframe df["50ma-200ma"], to make it easier
    to recognize when 50ma > 200ma
    '''
    df["50ma-200ma"] = df["50ma"] - df["200ma"]
    '''
    Then we use np.where to find, when df["50ma-200ma"]>0, is true. 
    The code is saying: when df["50ma-200ma"]>0, multiply it with one, otherwise
    multiply it with 0.
    '''
    df["where"] = np.where(df["50ma-200ma"]>0, 1, 0)
    '''
    Here we are doing the same thing as above. We are finding out when 
    df["50ma-200ma"] is less or equal to 0. By making this line, it gives
    us the option to go short, when we want to sell the company. 
    So if we truly believe in our strategy, we want to short every time we want to "sell"
    because we believe that the stock will go down. If we want to showcase that, 
    We simply changes the 0 in the middle to minus -1. So every time 50ma<200ma
    And the return is negative, we get a positvie return. But for now we dont go short,
    simply because we do not truly believe in our strategy.  
    '''
    df["where"] = np.where(df["50ma-200ma"]<=0, 0, 1)
    '''
    Here we are calculating the daily return of Tesla/Market.
    We are using pct_change, to calculate the percent, and then using 
    cumprod, to find the accumulated product of all returns. 
    '''
    df["Market cum"] = ((df["Adj Close"].pct_change()+1).cumprod()-1)
    df["Market"] = df["Adj Close"].pct_change()
    '''
    Here we are using the return of df["Market"] to calculate our return on 
    the strategy. The strategy is delayed by one day since it is calculated 
    with the closing price, thus we would only be able to react the day after the
    'buy' signal.
    '''
    df["Strategy"] = df["Market"] * df["where"].shift(1)
    df["Strategy cum"]=((df["Strategy"]+1).cumprod()-1)
    
    # scale up:
    df["Market cum"]=df["Market cum"]*100
    df["Market"] = df["Market"]*100
    df["Strategy"] = df["Strategy"]*100
    df["Strategy cum"] =df["Strategy cum"]*100
    
########## Part 4: How did our strategy do? ###################################
    
    
    '''
    We then plot the results of the market and our strategy. 
    '''
    plot2 = df[['Market cum','Strategy cum']].plot(
            grid=True, linestyle = "-", color=("black", "red"))
    
    
    '''Setting, labels, patches and title'''
    plot2.set_title("Strategy VS Market", size=30, color="Black", fontweight="bold")
    Market_label = mpatches.Patch(color="black", label= "Market")
    Strategy_label = mpatches.Patch(color="red", label="Strategy")
    Buying_label = mpatches.Patch(color="tab:purple", label="Buying")
    Selling_label = mpatches.Patch(color="tab:brown", label="Selling")
    plot2.legend(handles=[Market_label, Strategy_label, Buying_label, Selling_label], 
                 loc="upper left", prop={'size': 15}, frameon=False), plt.ylabel("Rate of return in percent"), plt.xticks(rotation=0)
    
    '''
    To clearly see, when we want sell and buy, we want to mark these days marked
    with horisontal lines.
    
    
    First we want to find the dates where we buy and sell. 
    In df["where"] there is only zeroes and ones. We know that when it is one, 
    we are holding the stock and 0 when we dont want to hold the stock. 
    By finding out when df["where"] goes from 0 to 1, and 1 to 0. We can
    find out when we want to buy and sell.
    
    
    First we make a new dataframe. We can use to specify when it goes from 0 to 1. 
    And 1 to 0. 
    '''
    df["Signal"] = np.sign(df["where"] - df["where"].shift(1))
    
    '''Then we make use of our new dataframe'''
    horizontal_lines_gold = df.loc[df["Signal"] == 1].index
    horizontal_lines_death = df.loc[df["Signal"] == -1].index
    '''
    Here we are using a for loop, to loop trough all the dates,
    and making horisontal lines at golden and death -cross. 
    '''
    for date in horizontal_lines_gold:
        plt.axvline(x=date, linestyle = "-", color="tab:purple")
    
    
    for date in horizontal_lines_death:
        plt.axvline(x=date, linestyle = "-", color="tab:brown")
    
    plt.xticks(rotation=0)
    return df

########## 5: Conclusion ##################################################### 
    
def figure_3(df):
    '''
    Our main goal is to beat the market. We therefore make a new plot, where 
    we find the different between our strategys return and the market/Teslas return. 
    '''
    
    df["Diff"] = df["Strategy"] - df["Market"]
    plot3 = df[['Diff']].cumsum().plot(grid=True, linestyle = "-", color=("red"))
    
    
    plot3.set_title("Difference in percentage points", size=30, color="Black", fontweight="bold")
    Percent_points_different_label = mpatches.Patch(color="Red", label= "Difference in percentage points")
    plot3.legend(handles=[Percent_points_different_label], 
                 loc="upper left", prop={'size': 15}, frameon=False), plt.ylabel("Percentage Points"), plt.xticks(rotation=0)
    '''
    The figure 3, is clearly showing that our basics strategy based on gold- and death cross, 
    is underperforming the market. Though, you should keep in mind, that this is only showing
    the results for one company. (Although we do not believe we found the way to El dorado). 
    '''
            