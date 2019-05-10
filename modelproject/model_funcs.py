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

    bounds = np.full((t,2),[0,1])
    res = optimize.minimize(obj, sks0, method='SLSQP', bounds=bounds)
    
    if res.success:
        if first:
            return res.x[0]
        else:
            return res.x  
    else:
        print('Optimization was sadly not succesfull')