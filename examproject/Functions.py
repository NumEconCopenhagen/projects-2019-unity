# Plotting specific imports:
from bokeh.io import output_notebook, push_notebook, show
from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource, HoverTool, NumeralTickFormatter, Label,Range1d
from bokeh.layouts import row, column
import ipywidgets as widgets
from IPython.display import display
import numpy as np


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
    tooltips=[(f'{x}','@x{0.00}')]
    
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
        tools=[hover,tools], x_axis_label=f'{x}', y_axis_label=f'{y_name}')

    for i,(yc,yn) in enumerate(zip(y_calls,y_names)):
        p.line(x='x', y=f'{yc}', source=source, legend=f'{yn}', 
        color = colors[i],line_width=line_width)
    

    p.legend.location = legendlocation
    
    return p 



def plotting_asad(x, x_array, y_arrays,y_name, equilibrium_path,title='Figure',
                colors= ['red','blue','green','purple','yellow'],y_range=None,
                legendlocation="top_center",tools="pan,wheel_zoom,box_zoom,reset,save",
                width=450, height=450, y_unit='',line_width=1.5,keep_ad1=False,color_ad1=(226, 123, 86)): 
    
    '''
    This function is made for plotting the as-ad model after a chock interactively,
    and its equilibrium path back to long term equilibrium.
    The function makes the plot, and also shows it, when called in a notebook

    Args:
            x(string)       : Name of x-axis
            x_array(array)  : Data for x-variable e.g. a linspace of outputgabs
            y_arrays(list)  : Containing lists with SRAS- and AD-curves-arrays
            y_name(string)  : Name of y_axis
            equilibrium_path (list) : of two arrays with the path of eqiulibrium for inflation- and output-gab.
            title(string)   : Figure title
            colors (list)   : list of colors of the lines, code for Hex colors is also accepted
            y_range(None or list) : If you wish decide the length of the y_axis this arg can be called with input [min,max]
            legendlocation(string): location of legend, written as "horizontalspace_verticalspace"
                                    Horizontalspace can be bottom  or center
                                    Verticalspace can be left, middle or right
            tools(string)       : Bokeh interactive tools, for the plot
            width,height (ints) : Determines the size of the figure
            y_unit(string)      : Unit of the y-axis data like % or $
            line_width (float)  : Width of the lines that are plotted
            keep_ad1 (bool)     : Wether to keep the AD-curve in the plot when another period is chosen interactively
            color_ad1 (string or hex-code) : The color of the AD-curve in period 1, if it is kept
    '''
    

    # Curves datasource:
    data = {'x': x_array,
            'ad_legend':[f'AD, t=0' for i in range(len(x_array))],
            'sras_legend': [f'SRAS, t=0' for i in range(len(x_array))],
            'sras-1':y_arrays[0][0],'ad-1':y_arrays[1][0],
            'sras':y_arrays[0][1],'ad':y_arrays[1][1]}


    if keep_ad1:
        data['ad0'] = y_arrays[1][1]
    
    source = ColumnDataSource(data)

    # Equilibirum datasource:
    eq_data = {'ys':equilibrium_path[0],'pis':equilibrium_path[1]}
    eq_source = ColumnDataSource(eq_data)


    # Figure and plot
    p = figure(plot_width=width, plot_height=height, title=f'{title}', 
        tools=tools, x_axis_label=f'{x}', y_axis_label=f'{y_name}')
    if y_range != None:
        p.y_range=Range1d(y_range[0], y_range[1])

    if keep_ad1:
        p.line(x='x', y='ad0', source=source, legend='AD, t=0',
           color =  color_ad1 ,line_width=line_width)  
    
    p.line(x='x', y='sras-1', source=source, legend='SRAS, t=-1',
           color = colors[0],line_width=line_width)
    p.line(x='x', y='ad-1', source=source, legend='AD, t=-1',
           color = colors[1],line_width=line_width)  
    
    p.line(x='x', y='sras', source=source, legend='sras_legend',
           color = colors[2],line_width=line_width, line_dash='dashed')
    
    p.line(x='x', y='ad', source=source, legend='ad_legend',
           color = colors[3],line_width=line_width, line_dash='dashed')
    

    p.circle(x='ys',y='pis',legend='Path of equilibrium',
            color=(106, 244, 65),source=eq_source)

    p.legend.location = legendlocation


    # Interactive function    
    def new_period(period):
        '''
        Updates the data of the plot the new period 
        '''
        data = {'x':x_array,'sras-1':y_arrays[0][0],'ad-1':y_arrays[1][0],
                'sras':y_arrays[0][period+1],'ad':y_arrays[1][period+1],
               'ad_legend':np.full(len(x_array),f'AD, t={period}'),
               'sras_legend': np.full(len(x_array),f'SRAS, t={period}')}
        if keep_ad1:
            data['ad0'] = y_arrays[1][1]
            
        source.data = ColumnDataSource(data).data
        
        push_notebook()
        
    
    slider = widgets.IntSlider(min=0,max=(len(y_arrays[1])-2),step=1,value=0)
    out = widgets.interactive(new_period,period=slider)
    display(out)

    show(p,notebook_handle=True)

def plot_hist(hist, edges, names=[''],tools="pan,wheel_zoom,box_zoom,reset,save",
             plot_range = False,x_label='x',y_label='y',title='Figure',
             alpha=0.5,legendlocation='top_right',width=500,height=500,
             fill_colors=['blue'],line_colors=['purple']):
    '''
    Plots a histogram using bokeh, the data is most easily be prepared using np.histogram()
    before being inputted into this function. 
    
    Args:
            hist  (list)        : Containing arrays with distribution of the data
            edges (list)        : Containing array with x-axis bins-location-data
            names (list)        : With names for the histograms if muliple are plotted
            tools (string)      : Bokeh tools
            plot_range(list)    : If you wish to decide the range of the x-axis, 
                this argument can be called as a list with: [min,max]
            x_label(string)     : Label of the x-axis
            y_label(string)     : Label of the y-axis
            title(string)       : Title of the figure
            fill_colors(list)   : Color(s) to fill the histogram(s), hex-color-code is also accepted
            line_colors(list)   : Color(s) in the line surrounding the histogram(s)
    
    Returns:
            p (bokeh.plotting.figure.Figure): The figure, which has to be called 
                in the bokeh.plotting comand, show(), to be viewed
    

    '''
    p = figure(title=title, tools=tools, x_axis_label=x_label, 
               y_axis_label=y_label,plot_width=width, plot_height=height)
    
    
    for h,e,name,fill_color,line_color in zip(hist,edges,names,fill_colors,line_colors):
        p.quad(top=h, bottom=0, left=e[:-1], right=e[1:],
               fill_color=fill_color, line_color=line_color, alpha=alpha,
              legend=name)
    
    p.y_range.start=0
    if plot_range == False:
        p.x_range.start = edges[0][0]
        p.x_range.end = edges[0][-1]
    else:
        p.x_range.start = plot_range[0]
        p.x_range.end = plot_range[-1]
    
    if names != ['']:
        p.legend.location = legendlocation
    
    return p

def utility(x1,x2,x3,beta1,beta2,beta3, gamma):
    utility = (x1**beta1*x2**beta2*x3**beta3)**gamma
    
    return utility


def utility_distribution(x1s,x2s,x3s,x1s_equal,x2s_equal,x3s_equal,betas,gamma,plot_range=[0,4]):
    '''
    Calculates the distribution of utility for all comsumer, for a given gamma and for two levels of comsumption for all comsumers,
    one derived from randomly distributed endowments, and one for equally distributed endowments
    Calculates the mean and variance, and makes a two figures containing everything

    Args:
            x1s (array)        : Comsumption of good 1 for each comsumer
            x2s (array)        : Comsumption of good 2 for each comsumer
            x3s (array)        : Comsumption of good 3 for each comsumer
            x1s_equal (array)  : Comsumption of good 1 for each comsumer (Equal distribution of endowments)
            x2s_equal (array)  : Comsumption of good 2 for each comsumer (Equal distribution of endowments)
            x2s_equal (array)  : Comsumption of good 3 for each comsumer (Equal distribution of endowments)
            betas (array)      : Containing beta for all comsumers for all goods
            gamma (float)      : Parameter
            plot_range(list)   : Containing min and max of range of the plotted x-axis. 
    Returns:
            plot1 (bokeh.plotting.figure.Figure) : The figure, for random endowments, which has to be called 
                in the bokeh.plotting comand, show(), to be viewed
            plot2 (bokeh.plotting.figure.Figure) : The figure, for equal endowments, which has to be called 
                in the bokeh.plotting comand, show(), to be viewed
    '''

    # Random endowments
    utilitys = []
    for i in range(len(x1s)):
        utilitys.append(utility(x1s[i],x2s[i],x3s[i],betas[i,0],betas[i,1],betas[i,2], gamma))
    
    hist, edges = np.histogram(utilitys, bins=150)
    plot1 = plot_hist([hist], [edges],names= [''],plot_range=plot_range,
            y_label='Observations',x_label='Utility',
            title=f'Randomly distributed endowments, gamma = {gamma:.2f}',
            width=500,height=350)
    
    mean = np.mean(utilitys)
    variance = np.var(utilitys)

    meantext = Label(x=250, y=215, text=f'Mean       = {mean:.4f}',
              text_font_size='10pt',x_units='screen', y_units='screen')
    vartext = Label(x=250, y=200, text=f'Variance  = {variance:.4f}',
              text_font_size='10pt',x_units='screen', y_units='screen')
    plot1.add_layout(meantext)
    plot1.add_layout(vartext)
    
    # Equal endowments
    utilitys_equal = []
    for i in range(len(x1s_equal)):
        utilitys_equal.append(utility(x1s_equal[i],x2s_equal[i],x3s_equal[i],betas[i,0],betas[i,1],betas[i,2], gamma))

    
    hist, edges = np.histogram(utilitys_equal, bins=150)
    plot2 = plot_hist([hist], [edges],names= [''],plot_range=plot_range,
                y_label='Observations',x_label='Utility',
                title=f'Equally distributed endowments, gamma = {gamma:.2f}',
                width=500,height=350)
    
    mean = np.mean(utilitys_equal)
    variance = np.var(utilitys_equal)

    meantext = Label(x=250, y=215, text=f'Mean       = {mean:.4f}',
              text_font_size='10pt',x_units='screen', y_units='screen')
    vartext = Label(x=250, y=200, text=f'Variance  = {variance:.4f}',
              text_font_size='10pt',x_units='screen', y_units='screen')
    plot2.add_layout(meantext)
    plot2.add_layout(vartext)
    
    
    return plot1, plot2