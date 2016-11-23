import numpy as np
import data.data_gen_scripts as data_gen

class LinRegr():
    def __init__(self):
        self.theta = None
        self.theta0 = None
        
    def train(self, X, y, feat_map=None):
        X_e = np.copy(X)
        X_m = np.copy(X)
        
        if feat_map:
            print 'feature map detected'
            X_m = np.array(map(feat_map, X_e))
            X_e = np.copy(X_m)
        
        X_e = np.array(map(lambda el: np.concatenate((el, [1.0])), X_e))

        n = len(X_e)

        try:
            res = np.dot(np.linalg.inv(np.dot(X_e.T,X_e)), np.dot(X_e.T, y))
            self.theta0 = res[len(res) - 1]
            self.theta = res[0:(len(res) - 1)]
        except np.linalg.LinAlgError:
            print 'ERROR: Extended X mx is not invertable'
            return
        
        print 'theta:', self.theta
        print 'theta0:', self.theta0, '\n'
        
        var = sum(map(lambda a: a**2, [ (y[i] - np.dot(self.theta, X_m[i]) - self.theta0) for i in range(n) ]))/n
        
        print 'variance:', var
        
    def avg_pred_err(self, X, y_r):
        sample_len = len(X)
        err = sum([ y_r[i] - np.dot(self.theta, X[i]) - self.theta0 for i in range(sample_len) ])/sample_len
        print 'average prediction error:', err, '\n'
        print '########################################', '\n'

        
theta_r = [1,2,1]
theta0_r = 0.5

X = data_gen.genX(2000, 3)    
y_r, y = data_gen.genYn(X, theta_r, theta0_r)
#print y_r.shape, y.shape


lrm0 = LinRegr()
lrm1 = LinRegr()
lrm2 = LinRegr()

def feat_map(row):
    return [1, row[0], row[1], row[2]]
    
def feat_map2(row):
    return [1, np.log(row[0]**2), np.log(row[1]**2), np.log(row[2]**2)]

print '########################################', '\n'

lrm0.train(X, y)
lrm0.avg_pred_err(X, y_r)

lrm1.train(X, y, feat_map)
lrm1.avg_pred_err(X, y_r)

# lrm2.train(X, y, feat_map2)
# lrm2.avg_pred_err(X, y_r)


print lrm0.theta, lrm1.theta