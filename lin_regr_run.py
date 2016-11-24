import data.data_gen_scripts as data_gen
from lib.LinRegr import LinRegr

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