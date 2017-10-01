# -*- coding: utf-8 -*-
#
# Regression analysis by using error function (least square method)
# 誤差関数（最小二乗法）を使用した回帰分析

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series, DataFrame

from numpy.random import normal

#------------
# Parameter
N=10           # number of x which gets samples
M=[0,1,3,9]    # list of degree of a polynominal

#------------
# prepare a dataset {x_n,y_n} (n=1...N) 
# (x, y) = (x, sign(2πx) + Randam number with standard deviation=0.3)
def create_dataset(num):
    dataset = DataFrame(columns=['x','y'])
    for i in range(num):
        x = float(i)/float(num-1)
        y = np.sin(2*np.pi*x) + normal(scale=0.3)
        dataset = dataset.append(Series([x,y], index=['x','y']),
                                 ignore_index=True)
    return dataset

#------------
# Calculate the Root mean square error
# 平方根平均二乗誤差
def rms_error(dataset, f):
    err = 0.0
    for index, line in dataset.iterrows():
        x, y = line.x, line.y
        err += 0.5 * (y - f(x))**2
    return np.sqrt(2 * err / len(dataset))

#------------
# Find a solution by using least square method.
def resolve(dataset, m):
    t = dataset.y
    phi = DataFrame()
    for i in range(0, m+1):
        p = dataset.x**i
        p.name="x**%d" % i
        phi = pd.concat([phi,p], axis=1)
    tmp = np.linalg.inv(np.dot(phi.T, phi))
    ws = np.dot(np.dot(tmp, phi.T), t)

    def f(x):
        y = 0
        for i, w in enumerate(ws):
            y += w * (x ** i)
        return y

    return (f, ws)

# Main
if __name__ == '__main__':
    train_set = create_dataset(N)
    print('--- train_set')
    print(train_set)
    
    test_set = create_dataset(N)
    print('--- train_set')
    print(test_set)

    df_ws = DataFrame()
    fig = plt.figure()
    
    # enumerate- Iterate over indices and items of a list
    for c, m in enumerate(M):
        
        # Find a solution by using least square method in training set.
        f, ws = resolve(train_set, m)
        df_ws = df_ws.append(Series(ws,name="M=%d" % m))
        
        # Set plot environment --> subplot(nrows, ncols, plot_number)
        # .add_subplot(2,2,2)
        #   - Add a plot at (1,2) position in 2 x 2 grid
        subplot = fig.add_subplot(2,2,c+1)
        subplot.set_xlim(-0.05,1.05)
        subplot.set_ylim(-1.5,1.5)
        subplot.set_title("M=%d" % m)

        # Plot train_set as scatter.
        subplot.scatter(train_set.x, train_set.y,
                        marker='o', color='blue', label=None)

        # Plot curb as line. y = sign(2πx)
        # .linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)
        #   - Return evenly spaced numbers over a specified interval.
        #   - num : Number of samples to generate. Must be non-negative.
        linex = np.linspace(0,1,101)
        liney = np.sin(2*np.pi*linex)
        subplot.plot(linex, liney, color='green', linestyle='--')

        # Plot polynominal approximate curve.
        linex = np.linspace(0,1,101)
        liney = f(linex)
        
        label = "E(RMS)=%.2f" % rms_error(train_set, f)
        subplot.plot(linex, liney, color='red', label=label)
        subplot.legend(loc=1)

    # Table of the coefficients
    print ("Table of the coefficients")
    print (df_ws.transpose())
    fig.show()

    # Display the changing of Root Mean Square Error for Training Set and Test Set.
    df = DataFrame(columns=['Training set','Test set'])
    for m in range(0,10):   # Degree of polynomiral as x axis on the graph.
        f, ws = resolve(train_set, m)
        train_error = rms_error(train_set, f)
        test_error = rms_error(test_set, f)
        df = df.append(
                Series([train_error, test_error],
                    index=['Training set','Test set']),
                ignore_index=True)
    df.plot(title='RMS Error', style=['-','--'], grid=True, ylim=(0,0.9))
    plt.show()
