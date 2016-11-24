import numpy as np

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
