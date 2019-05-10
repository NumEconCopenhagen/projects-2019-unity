import sympy as sm

def steadystate():
    
    sk = sm.symbols('s_Kt')
    alpha = sm.symbols('alpha')
    k =sm.symbols('k^{*}')
    delta = sm.symbols('delta')
    n = sm.symbols('n')
    b = sm.symbols('B')
    sseq = sm.Eq(k,1/(1+n)*(sk*b*k**alpha+(1-delta)*k))
    ss_k = sm.solve(sseq,k)[0]
    sm.Eq(k,ss_k)
    
    find_ssk_sk = sm.lambdify((k,b,delta,n,alpha),sm.solve(sseq,sk)[0])
    return sm.Eq(k,ss_k),find_ssk_sk


from scipy import optimize
import numpy as np

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


# Macro, the solow walk
def capitalakku(b,k,l,sk,alpha,delta):
    return prod(k,l,alpha,b)*sk+(1-delta)*k

def mod_solowwalk(k0, b, l0, n, alpha, delta, weight, theta, t, t_big):
    k_path = np.array([k0])
    l_path = np.array([l0*(1+n)**i for i in list(range(t_big))])
    y_path = np.array([prod(k_path[0],l_path[0],alpha,b)])
    
    sk_path = np.array([optimal_sks2(
        t, b, n, weight, delta, alpha, theta, k_path[0],l_path[0])])
    
    for i in range(1,t_big):
        k_plus = capitalakku(b,k_path[i-1],l_path[i-1],sk_path[i-1],alpha,delta)
        y_plus = prod(k_plus,l_path[i-1],alpha,b)
        sk_plus = np.array([optimal_sks2(t, b, n, weight, delta, alpha, theta, k_plus,l_path[i])])
        
        k_path = np.append(k_path, k_plus)
        y_path = np.append(y_path, y_plus)
        sk_path = np.append(sk_path,sk_plus)
        
    k_pr_path = k_path/l_path   
    y_pr_path = y_path/l_path                      
    return k_pr_path, y_pr_path, sk_path




# Plotting function:
from bokeh.io import output_notebook, push_notebook,show
from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource, HoverTool, NumeralTickFormatter

def plotting(x,y_names, x_array, y_arrays,y_calls,y_name ='Savings rate',title = 'Figure'
            , colors= ['red','blue','green','yellow'],legendlocation="top_center"): 
    

    tools="pan,box_zoom,reset,save"
    tooltips=[(f'{x}','@x{0,0.00}')]
    
    for yn,yc in zip(y_names,y_calls):
        text = '@'+f'{yc}'+'{0.00}'
        tooltips.append((f'{y_name} for {yn}',text))
    
    hover = HoverTool(tooltips=tooltips)
    data = {'x': x_array}
    for yc, y_array in zip(y_calls,y_arrays):
        data[f'{yc}']=y_array
        
    source = ColumnDataSource(data)
    
    p = figure(title=f'{title}',tools=[hover,tools], 
               x_axis_label=f'{x}', y_axis_label=f'{y_name}')
    for i,(yc,yn) in enumerate(zip(y_calls,y_names)):
        p.line(x='x', y=f'{yc}', source=source, 
           legend= f'{yn}', color = colors[i])
    
    p.legend.location = legendlocation
    
    show(p,notebook_handle=True)
    