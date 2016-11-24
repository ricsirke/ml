import numpy as np
import scipy as scp

def save(y_r, y_n):
    filename = raw_input('Please enter the new data file name!\n')
    with open(filename + "_n.dat", 'w') as fi:
        for el in y_n:
            fi.write(str(el) + "\n")

    with open(filename + "_r.dat", 'w') as fi:
            for el in y_r:
                fi.write(str(el) + "\n")






def y_func(x):
    return x**2


def gen__():
    sample_len = 2000
    nx = 40
    dx = 1/float(nx)
    x = [i*dx for i in range(sample_len)]
    y_r = map(y_func, x)
    noise = np.random.randn(sample_len) 
    y_n = [ y_r[i] + noise[i] for i in range(len(y_r)) ]   
    return y_r, y_n
 
def gen3d():
    y3r = []
    y3n = []
    for i in range(3):
        yr, yn = gen()
        y3r.append(yr)
        y3n.append(yn)
        
    return np.array(y3r).T, np.array(y3n).T
    
    
def genX(sample_len, dim):
    return np.random.randn(sample_len, dim)


def genYn(X, theta, theta0):
    dim = X.shape[1]
    sample_len = X.shape[0]
        
    noises = np.random.randn(sample_len)
    
    y_r = np.array(map(lambda e: np.dot(theta, e) + theta0, X))
    y_n = np.array([noises[i] + y_r[i] for i in range(sample_len)])
    
    return y_r, y_n
    

