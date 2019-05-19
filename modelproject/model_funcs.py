import sympy as sm
import numpy as np
from scipy import optimize
from scipy import interpolate


## Micro optimization
def total_utility(c, weight, theta):
    '''
    Sums utility for c for multiple years
    c is an array 
    '''
    if theta == 1:
        uts = np.log(c)
    else:
        uts = (c**(1-theta)-1)/(1-theta)

    # sum of utitity
    t_u = np.dot(uts,weight)

    return t_u

def prod(k,l,alpha,a):
    return (k**alpha)*((a*l)**(1-alpha))

def tot_ut_multiple_sks_quick(sks, k0, l, a, weight, theta, alpha, delta):
    '''
    Finds total utitilty for a set of years with a savingsrate for each year
    '''
    t = len(sks)
    k_short = np.empty(t)
    k_short[0] = k0
    
    for i in range(1,t):    
        k_short[i]=sks[i-1]*prod(k_short[i-1],l[i-1],alpha,a)+(1-delta)*k_short[i-1]
    
    y_short = prod(k_short,l,alpha,a)
    
    return total_utility(y_short*(1-sks)/l, weight, theta)


def optimal_sks(t, a, l, weight, delta, alpha, theta, k0, first=True):
    
    '''Optimizes the utility of the representative household
    by simulating the predicted future
    '''

    obj = lambda sks: -tot_ut_multiple_sks_quick(sks, k0, l, a, weight, theta, alpha, delta)
    sks0 = np.linspace(alpha,0,t)

    bounds = np.full((t,2),[1e-8,0.99999])
    res = optimize.minimize(obj, sks0, method='SLSQP', 
        bounds=bounds,)
    
    if res.success:
        if first:
            return res.x[0]
        else:
            return res.x  
    else:
        print('Optimization was sadly not succesfull')

def optimal_sks2(t, a, n, weight, delta, alpha, theta, k0, l0):
    '''
    This funcition also solves the maximization problem but instead of taking 
    a vector of populations growth, it only take l0 and calculates the rest
    this is for use in the simulation of the macro model, since the population is changing in every period. 
    '''
    l = np.array([l0*(1+n)**i for i in range(t)])
    obj = lambda sks: -tot_ut_multiple_sks_quick(sks, k0, l, a, weight, theta, alpha, delta)
    sks0 = np.linspace(alpha,0,t)

    bounds = np.full((t,2),[1e-8,0.99999])
    res = optimize.minimize(obj, sks0, method='SLSQP', 
        bounds=bounds,)
    
    if res.success: 
        return res.x[0]
    else:
        print('Optimization was sadly not succesfull')

def solve_micro(t, a, n, weight, delta, alpha, theta,l0,k_min=1e-4, k_max=30, precision=150):
    sks = np.zeros(precision)
    k0s = np.linspace(k_min,k_max,precision)
    for i in range(precision):
            sks[i] = optimal_sks2(t, a, n, weight, delta, alpha, theta, k0s[i],l0)

    sk_interp = interpolate.RegularGridInterpolator([k0s], sks,
        bounds_error=False,fill_value=None)
    return sks, k0s, sk_interp

def comsumption_plan(t, a, n, beta, delta, alpha, theta, k0, l0):
    '''
    Solving the micro problem, returns the entire plan for t periods 

    Args:
        t (int)         : Number of periods evaluated by the representative household
        a (float)       : The A parameter in the production function
        n (float)       : Yearly population growth
        beta (float)    : Time discount factor of the representative household
        delta (float)   : Depreciation rate
        alpha (float)   : Alpha in the productionfunction
        theta (float)   : Parameter in the utitlty function
        k0 (float)      : Initial level of capital
        l0 (float)      : Inital population
    
    Returns:
        sks (np.array)      : The chosen savings rate in each period, timed by 100 to get percentage
        k_pr_plan(np.array) : Planned level of capital in each period 
        c_pr_plan (np.array): Planned level of consumption in each period.
    '''
    
    weight = np.array([beta**i for i in range(t)])
    l = np.array([l0*(1+n)**i for i in range(t)])
    sks = optimal_sks(t, a, l, weight, delta, alpha, theta, k0, first=False)
    
    k_pr_plan = np.array([k0])
    for i in range(1,t):
        k_pr_plus = transition_eq(a,k_pr_plan[i-1],n,sks[i-1],alpha,delta)
        k_pr_plan = np.append(k_pr_plan, k_pr_plus)
    
    c_pr_plan = (1-sks)*a*(k_pr_plan**alpha)
    
    return sks*100, k_pr_plan, c_pr_plan


## Macro, the solow walks ##
def capitalakku(a,k,l,sk,alpha,delta):
    return prod(k,l,alpha,a)*sk+(1-delta)*k

def transition_eq(a,k_pr,n,sk,alpha,delta):
    return (sk*a*k_pr**alpha+(1-delta)*k_pr)/(1+n)

def solowwalk(k0, a, l0, n, sk, alpha, delta, timeline):
    '''Simulates the tradtional solow model with a fixed savings rate
    '''

    k_path = np.array([k0])
    l_path = np.array([l0*(1+n)**i for i in range(timeline)])
    y_path = np.array([prod(k_path[0],l_path[0],alpha,a)])
    
    for i in range(1,timeline):
        k_plus = capitalakku(a,k_path[i-1],l_path[i-1],sk,alpha,delta)
        y_plus = prod(k_plus,l_path[i-1],alpha,a)
        
        k_path = np.append(k_path, k_plus)
        y_path = np.append(y_path, y_plus)
        
    k_pr_path = k_path/l_path   
    y_pr_path = y_path/l_path                      
    return k_pr_path, y_pr_path


def new_mod_solowwalk(k0, l0, a,n, alpha, delta, sk_interp, timeline):
    '''
    Simulates the modifed solow model. In each period, the representative household
    calculates the preffered savingsrate in that period and the next period is simulated
    using that savings rate.
    '''
    k_path = np.array([k0])
    l_path = np.array([l0*(1+n)**i for i in range(timeline)])
    y_path = np.array([prod(k_path[0],l_path[0],alpha,a)])
    
    sk_path = np.array(sk_interp([k_path[0]/l_path[0]]))
    
    for i in range(1,timeline):
        k_plus = capitalakku(a,k_path[i-1],l_path[i-1],sk_path[i-1],alpha,delta)
        y_plus = prod(k_plus,l_path[i-1],alpha,a)
        sk_plus = np.array(sk_interp([k_plus/l_path[i]]))
        
        k_path = np.append(k_path, k_plus)
        y_path = np.append(y_path, y_plus)
        sk_path = np.append(sk_path,sk_plus)
        
    k_pr_path = k_path/l_path   
    y_pr_path = y_path/l_path                      
    return k_pr_path, y_pr_path, sk_path


## steady state calculations ##

def find_ssk_sk(k,a,delta,n,alpha):
    return (k**(1-alpha)*(delta+n))/a

def find_ssk_k(sk,a,delta,n,alpha):
    return ((a*sk)/(delta+n))**(1/(1-alpha))

def steadystate():
    '''
    Using sympy to calculate steady state in solow-model
    '''

    sk = sm.symbols('s_K')
    alpha = sm.symbols('alpha')
    k =sm.symbols('k^{*}')
    delta = sm.symbols('delta')
    n = sm.symbols('n')
    a = sm.symbols('A')
    sseq = sm.Eq(k,1/(1+n)*(sk*a*k**alpha+(1-delta)*k))
    ss_k = sm.solve(sseq,k)[0]
    sm.Eq(k,ss_k)
    
    return sm.Eq(k,ss_k)


def find_ss(t, a, n, beta, delta, alpha, theta, bracket=[0.01,30]):
    '''
    Finding the steady state of the model by figuring out which mount of capital,
    that makes the optimal savings rate chosen by the consumer equal
    to the savings rate the implies that this amount of capital is the steady state. 
    '''
    weight = np.array([beta**i for i in range(t)])

    obj = lambda k_star: find_ssk_sk(k_star,a,delta,n,alpha)-optimal_sks2(
        t, a, n, weight, delta, alpha, theta, k_star,1)
    res = optimize.root_scalar(obj,method='brentq',bracket=bracket)
    if res.converged:
        k_star = res.root
        sk_star = find_ssk_sk(k_star,a,delta,n,alpha)
        return k_star,sk_star
    else:
        print('Convergence failed')

def new_find_ss(sk_interp, a, n, beta, delta, alpha, bracket):
    '''
    Finding the steady state of the model by figuring out which mount of capital,
    that makes the optimal savings rate chosen by the consumer equal
    to the savings rate the implies that this amount of capital is the steady state. 
    This uses interpolate.
    '''
    
    obj = lambda k_star: find_ssk_sk(k_star,a,delta,n,alpha)-sk_interp([k_star])[0]
    res = optimize.root_scalar(obj,method='brentq',bracket=bracket)
    if res.converged:
        k_star = res.root
        sk_star = find_ssk_sk(k_star,a,delta,n,alpha)
        return k_star,sk_star
    else:
        print('Convergence failed')

def find_ss_alt(a, n, beta, delta, alpha, theta, k0,l0):
    t = 200
    weight = np.array([beta**i for i in range(t)])
    l = l0*np.array([(1+n)**i for i in range(t)])
    
    sks = optimal_sks(t, a, l, weight, delta, alpha, theta, k0, first=False)
    ss_sk = False
    i = int(0.25*t)
    while i < int(0.75*t):
        if np.isclose(sks[i],sks[i+1]):
            ss_sk = sks[i]
            break

        i+=1

    if ss_sk==False:
        # We try again with a longer timerframe
        t = 400
        weight = np.array([beta**i for i in range(t)])
        l = l0*np.array([(1+n)**i for i in range(t)])
    
        sks = optimal_sks(t, a, l, weight, delta, alpha, theta, k0, first=False)
        i=0
        while i < t:
            if np.isclose(sks[i],sks[i+1]):
                ss_sk=sks[i]
                break
            
            i += 1 
    # We use the analytical solution to find the steady state of capital
    # for this savings rate:
    ss_k = find_ssk_k(ss_sk,a,delta,n,alpha)
    return ss_sk, ss_k





##  Plotting functions ## :

# Plotting specific imports:
from bokeh.io import output_notebook, push_notebook, show
from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource, HoverTool, NumeralTickFormatter
from bokeh.layouts import row, column
import ipywidgets as widgets
from IPython.display import display


def plotting(x,y_names,  x_array, y_arrays,y_name ='Savings rate', title='Figure',
                colors= ['red','blue','green','purple','yellow'],
                legendlocation="top_center",tools="pan,wheel_zoom,box_zoom,reset,save",
                width=450, height=450): 
    
    '''
    Bokeh plotting
    '''
    
     # Bokeh needs a name for the data that neither has spaces nor numbers
    # because we want the option to do this we define abitrairy calls via the alphabeth. 
    calls = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 
     'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    y_calls = []

    for i in range(len(y_names)):
        y_calls.append(calls[i])
    

    tooltips=[(f'{x}','@x{0,0.00}')]
    
    for yn,yc in zip(y_names,y_calls):
        text = '@'+f'{yc}'+'{0.00}'
        tooltips.append((f'{y_name} for {yn}',text))
    
    hover = HoverTool(tooltips=tooltips)
    data = {'x': x_array}
    for yc, y_array in zip(y_calls,y_arrays):
        data[f'{yc}']=y_array
        
    source = ColumnDataSource(data)
    
    p = figure(plot_width=width, plot_height=height, title=f'{title}', 
        tools=[hover,tools], x_axis_label=f'{x}', y_axis_label=f'{y_name}')

    for i,(yc,yn) in enumerate(zip(y_calls,y_names)):
        p.line(x='x', y=f'{yc}', source=source, 
           legend= f'{yn}', color = colors[i])
    
    p.legend.location = legendlocation
    
    return p 


def plotting_plan():
    '''
    Plots the comsumption plan of the representative household, e.g. it creates three plots:
    One for the savingsrate (sks), one for capital pr. capita(k_pr_plan), 
    and one for comsumption pr. capita (c_pr_plan).
    It is plotted interactively so different parameter values can be evaluated. 
    '''

    # initial plan is solved for and saved as data for bokeh plot: 
    sks, k_pr_plan, c_pr_plan = comsumption_plan(200, 1, 0.01, 0.99, 0.05, 0.33, 0.5, 8, 1)
    data = {'T' : np.array(range(200)),'sks' : sks, 'k_pr' : k_pr_plan,'c_pr': c_pr_plan}
    source = ColumnDataSource(data)


    # bokeh tooltips is for showing the values of the figure when hovering the mouse above them.
    tooltips =[[('T','@T'),('Savings rate','@sks{0.00} %')],[('T','@T'),
                ('Capital pr. capita','@k_pr{0.00}')],[('T','@T'),('Consumption pr. capita','@c_pr{0.00}')]]
    
    tools="pan,wheel_zoom,box_zoom,reset,save"

    
    titles = ['Savingsrate in %', 'Capita pr. capita', 'Consumption pr. capita']
    
    ps = []
    
    # The three plots are created:
    for title, tooltip in zip(titles,tooltips):
        ps.append(figure(plot_width=430, plot_height=430, title=title, 
            tools=[HoverTool(tooltips=tooltip),tools], x_axis_label='T', y_axis_label=title))

    for p,y in zip(ps,['sks','k_pr','c_pr']):
        p.line(x='T', y=y, source=source, color = 'blue')
    
    
    # Function for interaction:
    def update_parameters(t=200,theta=0.5,beta=0.99, 
                        delta=0.05,alpha=1/3, a=1,k0=5,l0=1,n=0.008):
        '''
        Takes the parameter values that the user can define interactively,
        and solves the consumption plan and updates the plots
        '''
        sks, k_pr_plan, c_pr_plan = comsumption_plan(
            t, a, n, beta, delta, alpha, theta, k0, l0)
        
        data = {'T' : np.array(range(t)),'sks' : sks, 
                'k_pr' : k_pr_plan,'c_pr': c_pr_plan}
        
        source.data = ColumnDataSource(data).data
    
        #jupyter specific.
        push_notebook()
    
    # The first slider for Time periods is created, since this has to be and interger.
    sliders = {'t' : widgets.IntSlider(min=10,max=500,step=10,value=200, 
                description='T',orientation ='vertical')}


    variables = ['theta','beta','delta','alpha','a','k0','l0','n']
    # Description for the sliders raw-strings are used to enable latex:
    descriptions = [r'\(\theta\)',r'\(\beta\)',r'\(\delta\)',r'\(\alpha\)',
                    r'\(A\)',r'\(K_{0}\)',r'\(L_{0}\)','\(n\)']
    
    # min,max,step, intial values for all remaning parameters:
    values = [[0.1,5,0.1,0.5],[0.9,1,0.01,0.99],[0.01,0.15,0.01,0.05],[0.1,0.9,0.1,0.33],
            [1,10,1,1],[1,25,1,8],[1,25,1,1],[0,0.1,0.01,0.01]]
    
    # Sliders for remaing parameters is created in loop and addded to the dictionary.
    for variable, value, description in zip(variables,values,descriptions):
        sliders[f'{variable}'] = (widgets.FloatSlider(
            min=value[0],max=value[1], step=value[2],value=value[3],
            description=description, orientation= 'vertical'))
    
    #horizontal box to have the sliders next to eachother
    box = widgets.HBox([v for v in sliders.values()])

    # linking the sliders to the the update function. 
    out = widgets.interactive_output(update_parameters, sliders)

    display(out,box)


    #Rember to print the bokeh-plots:
    show(row(column(ps[0],ps[2]),ps[1]),notebook_handle=True)

