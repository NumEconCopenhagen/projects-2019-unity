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

    uts = (c**(1-theta)-1)/(1-theta)
    # sum of utitity
    t_u = np.dot(uts,weight)

    return t_u

def prod(k,l,alpha,a):
    return (k**alpha)*((a*l)**(1-alpha))

def tot_ut_multiple_sks_quick(sks, k0, l, b, weight, theta, alpha, delta):
    '''
    Finds total utitilty for a set of years with a savingsrate for each year
    '''
    t = len(sks)
    k_short = np.empty(t)
    k_short[0] = k0
    
    for i in range(1,t):    
        k_short[i]=sks[i-1]*prod(k_short[i-1],l[i-1],alpha,b)+(1-delta)*k_short[i-1]
    
    y_short = prod(k_short,l,alpha,b)
    
    return total_utility(y_short*(1-sks)/l, weight, theta)


def optimal_sks(t, b, l, weight, delta, alpha, theta, k0, first=True):
    '''Optimizes the utility of the representative household
    by simulating the predicted future'''
    obj = lambda sks: -tot_ut_multiple_sks_quick(sks, k0, l, b, weight, theta, alpha, delta)
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

def optimal_sks2(t, b, n, weight, delta, alpha, theta, k0, l0):
    '''This funcition also solves the maximization problem but instead of taking 
    a vector of populations growth, it only take l0 and calculates the rest
    this is for use in the simulation of the macro model, since the population is changing in every period. 
    '''
    l = np.array([l0*(1+n)**i for i in range(t)])
    obj = lambda sks: -tot_ut_multiple_sks_quick(sks, k0, l, b, weight, theta, alpha, delta)
    sks0 = np.linspace(alpha,0,t)

    bounds = np.full((t,2),[1e-8,0.99999])
    res = optimize.minimize(obj, sks0, method='SLSQP', 
        bounds=bounds,)
    
    if res.success: 
        return res.x[0]
    else:
        print('Optimization was sadly not succesfull')

def solve_micro(t, b, n, weight, delta, alpha, theta,l0,k_min=1e-4, k_high=30, precision=150):
    sks = np.zeros(precision)
    k0s = np.linspace(k_min,k_high,precision)
    for i in range(precision):
            sks[i] = optimal_sks2(t, b, n, weight, delta, alpha, theta, k0s[i],l0)

    sk_interp = interpolate.RegularGridInterpolator([k0s], sks,
        bounds_error=False,fill_value=None)
    return sks, k0s, sk_interp


# Macro, the solow walk
def capitalakku(b,k,l,sk,alpha,delta):
    return prod(k,l,alpha,b)*sk+(1-delta)*k

def solowwalk(k0, b, l0, n, sk, alpha, delta, timeline):
    '''Simulates the tradtional solow model with a fixed savings rate
    '''

    k_path = np.array([k0])
    l_path = np.array([l0*(1+n)**i for i in list(range(timeline))])
    y_path = np.array([prod(k_path[0],l_path[0],alpha,b)])
    
    for i in range(1,timeline):
        k_plus = capitalakku(b,k_path[i-1],l_path[i-1],sk,alpha,delta)
        y_plus = prod(k_plus,l_path[i-1],alpha,b)
        
        k_path = np.append(k_path, k_plus)
        y_path = np.append(y_path, y_plus)
        
    k_pr_path = k_path/l_path   
    y_pr_path = y_path/l_path                      
    return k_pr_path, y_pr_path



def mod_solowwalk(k0, b, l0, n, alpha, delta, weight, theta, t, timeline):
    '''Simulates the modifed solow model. In each period, the representative household
    calculates the preffered savingsrate in that period and the next period is simulated
    using that savings rate.
    '''

    k_path = np.array([k0])
    l_path = np.array([l0*(1+n)**i for i in list(range(timeline))])
    y_path = np.array([prod(k_path[0],l_path[0],alpha,b)])
    
    sk_path = np.array([optimal_sks2(
        t, b, n, weight, delta, alpha, theta, k_path[0],l_path[0])])
    
    for i in range(1,timeline):
        k_plus = capitalakku(b,k_path[i-1],l_path[i-1],sk_path[i-1],alpha,delta)
        y_plus = prod(k_plus,l_path[i-1],alpha,b)
        sk_plus = np.array([optimal_sks2(t, b, n, weight, delta, alpha, theta, k_plus,l_path[i])])
        
        k_path = np.append(k_path, k_plus)
        y_path = np.append(y_path, y_plus)
        sk_path = np.append(sk_path,sk_plus)
        
    k_pr_path = k_path/l_path   
    y_pr_path = y_path/l_path                      
    return k_pr_path, y_pr_path, sk_path


def new_mod_solowwalk(k0, l0, b,n, alpha, delta, sk_interp, timeline):
    '''
    Simulates the modifed solow model. In each period, the representative household
    calculates the preffered savingsrate in that period and the next period is simulated
    using that savings rate.
    '''
    k_path = np.array([k0])
    l_path = np.array([l0*(1+n)**i for i in list(range(timeline))])
    y_path = np.array([prod(k_path[0],l_path[0],alpha,b)])
    
    sk_path = np.array(sk_interp([k_path[0]/l_path[0]]))
    
    for i in range(1,timeline):
        k_plus = capitalakku(b,k_path[i-1],l_path[i-1],sk_path[i-1],alpha,delta)
        y_plus = prod(k_plus,l_path[i-1],alpha,b)
        sk_plus = np.array(sk_interp([k_plus/l_path[i]]))
        
        k_path = np.append(k_path, k_plus)
        y_path = np.append(y_path, y_plus)
        sk_path = np.append(sk_path,sk_plus)
        
    k_pr_path = k_path/l_path   
    y_pr_path = y_path/l_path                      
    return k_pr_path, y_pr_path, sk_path


## steady state calculations

def find_ssk_sk(k,b,delta,n,alpha):
    return (k**(1-alpha)*(delta+n))/b

def find_ssk_k(sk,b,delta,n,alpha):
    return ((b*sk)/(delta+n))**(1/(1-alpha))

def steadystate():
    '''
    Using sympy to calculate steady state in solow-model
    '''

    sk = sm.symbols('s_K')
    alpha = sm.symbols('alpha')
    k =sm.symbols('k^{*}')
    delta = sm.symbols('delta')
    n = sm.symbols('n')
    b = sm.symbols('B')
    sseq = sm.Eq(k,1/(1+n)*(sk*b*k**alpha+(1-delta)*k))
    ss_k = sm.solve(sseq,k)[0]
    sm.Eq(k,ss_k)
    
    find_ssk_sk = sm.lambdify((k,b,delta,n,alpha),sm.solve(sseq,sk)[0])
    return sm.Eq(k,ss_k)


def find_ss(t, b, n, beta, delta, alpha, theta, bracket):
    '''
    Finding the steady state of the model by figuring out which mount of capital,
    that makes the optimal savings rate chosen by the consumer equal
    to the savings rate the implies that this amount of capital is the steady state. 
    '''
    weight = np.array([beta**i for i in range(t)])

    obj = lambda k_star: find_ssk_sk(k_star,b,delta,n,alpha)-optimal_sks2(
        t, b, n, weight, delta, alpha, theta, k_star,1)
    res = optimize.root_scalar(obj,method='brentq',bracket=bracket)
    if res.converged:
        k_star = res.root
        sk_star = find_ssk_sk(k_star,b,delta,n,alpha)
        return k_star,sk_star
    else:
        print('Convergence failed')

def new_find_ss(sk_interp, b, n, beta, delta, alpha, bracket):
    '''
    Finding the steady state of the model by figuring out which mount of capital,
    that makes the optimal savings rate chosen by the consumer equal
    to the savings rate the implies that this amount of capital is the steady state. 
    This uses interpolate.
    '''
    

    obj = lambda k_star: find_ssk_sk(k_star,b,delta,n,alpha)-sk_interp([k_star])[0]
    res = optimize.root_scalar(obj,method='brentq',bracket=bracket)
    if res.converged:
        k_star = res.root
        sk_star = find_ssk_sk(k_star,b,delta,n,alpha)
        return k_star,sk_star
    else:
        print('Convergence failed')


# Plotting function:
from bokeh.io import output_notebook, push_notebook,show
from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource, HoverTool, NumeralTickFormatter

def plotting(x,y_names,  x_array, y_arrays,y_name ='Savings rate', title='Figure',
                colors= ['red','blue','green','purple','yellow'],
                legendlocation="top_center",tools="pan,wheel_zoom,box_zoom,reset,save",
                width=400, height=500): 
    
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
