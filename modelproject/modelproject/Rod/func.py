import matplotlib.pyplot as plt
from scipy import optimize
import numpy as np
import sympy as sm

def plot(x,y):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(x,y)

def plot2(x,y1,y2):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(x,y1)
    ax.plot(x,y2)


# Micro optimization
def total_utility(c, beta, theta):
    '''
    Sums utility for c for multiple years
    c is an array 
    '''
    uts = (c**(1-theta)-1)/(1-theta)
     #time dicounting: 
    timeweights = [beta**i for i in range(len(c))]
    # sum of utitity
    t_u = np.dot(uts,timeweights)
    
    return t_u

def prod(k,l,alpha,b):
    return (b*k**alpha)*(l**(1-alpha))

def tot_ut_multiple_sks_quick(sks, k0, l0, n, b, beta, theta, alpha, delta):
    '''
    Finds total utitilty for a set of years with a savingsrate for each year
    '''
    t = len(sks)
    k_short = np.empty(t)
    k_short[0] = k0
    l_short = np.array([l0*(1+n)**i for i in range(t)])
    
    for i in range(1,t):    
        k_short[i]=sks[i-1]*prod(k_short[i-1],l_short[i-1],alpha,b)+(1-delta)*k_short[i-1]
    
    y_short = prod(k_short,l_short,alpha,b)
    
    return total_utility(y_short*(1-sks)/l_short, beta, theta)

def optimal_sks(t, b, l0, n, beta, delta, alpha, theta, k0, first=True):
    obj = lambda sks: -tot_ut_multiple_sks_quick(sks, k0, l0, n, b, beta, theta, alpha, delta)
    sks0 = np.linspace(alpha,0,t)

    bounds = np.full((t,2),[0,1])
    res = optimize.minimize(obj, sks0, method='SLSQP', bounds=bounds)
    if res.success == False:
        print('Optimization was sadly not succesfull')
    elif first:
        return res.x[0]
    else:
        return res.x

