from Perceptron import Perceptron
from utils import *


X = load_data('data/p1_a_X.dat')
y = load_data('data/p1_a_y.dat')
print '#######################################', '\n'
model = Perceptron([0, 0])
model.train(X, y)
model.test(X, y)
model.info(X, y, [1, 0])
model.plot(X, y)
print '#######################################', '\n'

########################################

X2 = load_data('data/p1_b_X.dat')
y2 = load_data('data/p1_b_y.dat')
print '#######################################', '\n'
model2 = Perceptron([0, 0])
model2.train(X2, y2)
model2.test(X2, y2)
model2.info(X2, y2, [1, 0])
model2.plot(X2, y2)
print '#######################################', '\n'