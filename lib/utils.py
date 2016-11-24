import math
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import numpy as np

def load_data(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip().split()
            if len(line) == 1:
                data.append(float(line[0]))
            else:
                data.append(map(float, line))
    print 'read', '"' + filename + '"', 'of length:', len(data), '\n'
    return data

def dot_prod(x, y):
    return sum([float(i)*float(j) for (i, j) in zip(x, y)])
    
def norm(x):
    return math.sqrt(sum(map(lambda y: float(y**2), x)))
    
def normalize(x):
    len = norm(x)
    try:
        return map(lambda y: y/len, x)
    except ZeroDivisionError:
        return 0
    
def angle(x, y):
    try:
        return dot_prod(x, y)/( norm(x)*norm(y) )
    except ZeroDivisionError:
        return 0
        
def angle_deg(x, y):
    return angle(x, y)*180/math.pi
    
def dist_point_line(theta, x):
    return abs((dot_prod(theta, x))/(norm(theta)))
    
def plot(X, y, theta, theta0=0, margin=False):
    pos = np.array([ X[i] for i in range(len(X)) if y[i] == 1 ]).T
    neg = np.array([ X[i] for i in range(len(X)) if y[i] == -1 ]).T
    
    max_x = max(np.max(pos[1]), np.max(neg[1]))
    max_y = max(np.max(pos[0]), np.max(neg[0]))
    min_x = min(np.min(pos[1]), np.min(neg[1]))
    min_y = min(np.min(pos[0]), np.min(neg[0]))
    
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # indices in reverse order because of the transposition
    ax.scatter(pos[1], pos[0], c='r', marker='o')
    ax.scatter(neg[1], neg[0], c='b', marker='o')
    
    # decision boundary
    def line_y(line_x):
        line_slope = -theta[0]/theta[1]
        return line_slope*line_x - (theta0/theta[1])
    
    st_pt = (min_x, line_y(min_x))
    end_pt = (max_x, line_y(max_x))
    
    (line_xs, line_ys) = zip(*[st_pt, end_pt])
    
    ax.add_line(lines.Line2D(line_xs, line_ys, linewidth=2, color='black'))
    
    if margin:
        m1_st_pt = (min_x, line_y(min_x) + margin)
        m1_end_pt = (max_x, line_y(max_x) + margin)
        m2_st_pt = (min_x, line_y(min_x) - margin)
        m2_end_pt = (max_x, line_y(max_x) - margin)
        
        (m1_xs, m1_ys) = zip(*[m1_st_pt, m1_end_pt])
        (m2_xs, m2_ys) = zip(*[m2_st_pt, m2_end_pt])
        
        ax.add_line(lines.Line2D(m1_xs, m1_ys, linewidth=1, linestyle='dashed', color='black'))
        ax.add_line(lines.Line2D(m2_xs, m2_ys, linewidth=1, linestyle='dashed', color='black'))


    plt.plot()
    plt.show()