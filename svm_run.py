from SVM import SVM
from utils import *
import numpy as np

X = np.array(load_data('data/p1_a_X.dat'))
y = np.array(load_data('data/p1_a_y.dat'))

X2 = np.array(load_data('data/p1_b_X.dat'))
y2 = np.array(load_data('data/p1_b_y.dat'))

theta_start = [0, 0]
theta0 = 0

svm = SVM(theta_start, theta0)
svm.train(X, y)
svm.test(X, y)
svm.plot(X, y)

svm2 = SVM(theta_start, theta0)
svm2.train(X2, y2)
svm2.test(X2, y2)
svm2.plot(X2, y2)