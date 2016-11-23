#! /usr/bin/env python

# max gamma / norm(theta)                maximizing the margin
#       ^
#      ||
#      v
# min [ (1/2) * norm(theta)^2 ]
#
#   st    y( theta*x + theta0 ) >= 1     for all (y,x) from the training set            it was divided by gamma to get 1 on the right
#


######################################
## TODO: RELAXATION, REGULARIZATION ##     from lecture notes 3, 4
######################################


import numpy as np
from scipy import optimize
from utils import *

class SVM():
    def __init__(self, theta, theta0):
        self.theta0 = theta0
        self.theta = theta
        
    def train(self, X, y):
        H = np.identity(len(X[0])) # TODO: hardcoded
        A = np.array([ y[i]*X[i] for i in range(len(X)) ])*(-1)
        b = np.array([ 1-y[i]*self.theta0 for i in range(len(X)) ])*(-1)

        def loss(x, sign=1.):
            return sign * ( 0.5 * np.dot(x.T, np.dot(H, x)) )
                
        cons = {
            'type': 'ineq',
            'fun': lambda x: b - np.dot(A, x)
        }
                                         
        res_cons = optimize.minimize(loss, self.theta, constraints=cons, method='SLSQP', options={'disp': False})

        theta = res_cons.x
        margin = 1/norm(theta)
        max_val = 1/res_cons.fun
        
        print 'TRAIN:'
        print 'parameter:', theta
        print 'margin:', margin
        print 'max_val:', max_val, '\n'
        self.theta = theta
        
    def test(self, X, y):
        test_err = 0
        for i in range(len(X)):
            y_pred = np.sign(np.dot(self.theta, X[i]) + self.theta0)
            if y[i] != y_pred:
                test_err += 1
                
        print 'TEST:'
        print 'number of errors:', test_err, '\n'
        
    def plot(self, X, y):
        plot(X, y, self.theta, theta0=self.theta0, margin=(1/norm(self.theta)))


# TODO: use jacobian!!!!!

# def jacobian(x, sign=1.):
    # return sign * (np.dot(x.T, H))

# cons = {'type':'ineq',
        # 'fun':lambda x: b - np.dot(A,x),
        # 'jac':lambda x: -A}
        
# res_cons = optimize.minimize(loss, x0, jac=jacobian,constraints=cons,
                             # method='SLSQP', options=opt)
    