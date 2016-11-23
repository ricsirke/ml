#! /usr/bin/env python

from utils import *

class Perceptron:
    def __init__(self, theta):
        self.theta = theta
    
    def classify(self, x):
        arg = dot_prod(self.theta, x)
        if arg > 0:
            return 1
        elif arg == 0:
            return 0
        else:
            return -1

    def train(self,X,y):
        k = 0
        step = 0
        for i in range(len(X)):
            y_calc = self.classify(X[i])
            if y_calc != y[i]:
                k += 1
                step = i
                self.theta = [x1 + x2 for (x1, x2) in zip(self.theta, map(lambda x: x*y[i], X[i]))]
        
        print 'TRAINING:'
        print 'parameter:', self.theta
        print 'number of parameter updates:', k
        print 'number of steps before convergence:', step, '\n'
        
        return [self.theta, k, step]

    def test(self, X, y):        
        test_err = 0
        for i in range(len(X)):
            y_calc = self.classify(X[i])
            if y_calc != y[i]:
                test_err += 1
        
        print 'TEST:'
        print 'number of errors made:', test_err, '\n'
        return test_err

    def calc_gamma(self, X, y):
        # y*theta_opt*x >= gamma
        # we search for the minimum of the left hand side
        gamma = dot_prod(self.theta, X[0])*y[0]
        x = X[0]
        for i in range(1, len(X)):
            val = dot_prod(self.theta, X[i])*y[i]
            if val < gamma:
                gamma = val
                x = X[i]
                       
        print 'gamma:', gamma
        print 'closest point:', x
        print 'minimal distance:', dist_point_line(self.theta, x)
        return [gamma, x]
        
    def calc_R(self, X):
        R = norm(X[0])
        for i in range(1, len(X)):
            curr = norm(X[i])
            if R < curr:
                R = curr
        print 'R:', R
        return R
        
    def calc_angle(self, vec):
        angle = angle_deg(self.theta, vec)
        print 'angle between the parameter and', str(vec) +':', angle, 'degrees\n'       
        return angle
        
    def info(self, X, y, angle_vec):
        print 'INFO:'
        self.calc_gamma(X, y)
        self.calc_R(X)
        self.calc_angle(angle_vec)
        
    def plot(self, X, y):
        plot(X, y, self.theta, 0)
        
        
