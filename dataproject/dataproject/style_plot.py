import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
from matplotlib.transforms import Bbox
from matplotlib.path import Path
from matplotlib import style 
import matplotlib.dates
import matplotlib.animation as animation 
from mpl_finance import candlestick_ohlc
import mpl_finance 
import datetime as dt
import pandas_datareader.data as web
import pandas as pd
from matplotlib.ticker import FuncFormatter




def style_plot(from_year = 2015, to_year = 2018):

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
    start = dt.datetime(from_year, 1, 1)
    end = dt.datetime(to_year, 1, 1)
    df = web.DataReader(['TSLA'], 'yahoo', start, end)
    
    # Specifying a format for date to match our data
    df.index = pd.to_datetime(df.index)
    df['year']= df.index.year
    df.rename(columns={'Symbols':'Company'},inplace=True)
    
    # Due to weird dates in matplotlib, we are changing them
    df["Date"]=matplotlib.dates.date2num(df.index.to_pydatetime())
    
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
    candle_patch_red = mpatches.Patch(color="red", label= "Tesla adj. closing stock price, Open < Close")
    candle_patch__green = mpatches.Patch(color="green", label="Tesla adj. closing stock price, Open > Close")
    gold_patch = mpatches.Patch(color="maroon", label="200ma")
    silver_patch = mpatches.Patch(color="teal", label="50ma")
    volume_patch = mpatches.Patch(color="black", label="Volume")
    ax1.legend(handles=[ candle_patch__green, candle_patch_red,
                        gold_patch, silver_patch,
                        volume_patch], 
    loc="upper left", prop={'size': 15}, frameon=False)
    
    
    # Making a title for the graph. 
    ax1.set_title("Analyzing Tesla's stock price", size=30, color="Black", fontweight="bold")
    
    # Here we are using -annotate to making a arrow and text in the graph. 
    # We did set the cordinates for the text manually.
    ax1.annotate('Death Cross', size=11, fontweight="bold",  xy=(dt.date(2015, 11, 15), 200), 
                 xytext=(dt.date(2015, 9, 15), 150),
                arrowprops=dict(facecolor='black'),
                )
    ax1.annotate('Golden Cross', size=11, fontweight="bold", xy=(dt.date(2017, 2, 1), 200), 
                 xytext=(dt.date(2016, 12, 25), 150),
                arrowprops=dict(facecolor='black'),
                )
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
    plt.savefig("Golden Cross.png", facecolor=fig.get_facecolor(), edgecolor="b", dpi=300)

   
