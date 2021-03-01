
from __future__ import print_function
import sys
import re
import numpy as np
from numpy import dot
from numpy.linalg import norm
from operator import add
from pyspark import SparkContext
import pandas as pd


sc = SparkContext(appName="vector Gradient Descent")

        

lines = sc.textFile(sys.argv[1])

taxilines = lines.map(lambda X: X.split(','))

def isfloat (value):
    try:
        float(value)
        return True
    except:
        return False

def correctRows(p):
    if(len(p) == 17):
        if(isfloat(p[5])) and isfloat(p[11]):
            if(float(p[5]!=0) and float(p[1]!=0)):
                if(int(p[4])>=120) and (int (p[4]) <= 3600):
                    if(float(p[5]) >=1) and (float(p[5])<=50):
                        if(float(p[11]) >=3) and (float(p[11])<=200):
                            if float(p[15])>=3:
                                return p
def get_variable(p):
    return (p[5],p[11],float(p[5])**2,float(p[5])*float(p[11]))



def get_vector(p):
    x_array = np.array([float(p[4]),float(p[5]),float(p[11]),float(p[12])])
    y_minus_yp = (float(p[16]) - (np.dot(x_array,parameter_current[1:5])+parameter_current[0]))
    x_times_y = []
    for i in x_array:
        x_times_y.append(y_minus_yp*i)
    x_times_y_vector = np.array(x_times_y)
    return (float(y_minus_yp),x_array,x_times_y_vector,float(y_minus_yp )**2)





taxilinesCorrected = taxilines.filter(correctRows)

parameter_current = np.array([0.1,0.1,0.1,0.1,0.1])
precision = 0.1
learningRate = 0.001
num_iteration = 100
oldCost=0
previous_step_size = 1

n=taxilinesCorrected.count()

learningRate = 0.001
num_iteration = 100

parameter_vector_current = np.array([0.1,0.1,0.1,0.1,0.1])

taxilinesCorrected.cache()


for i in range(num_iteration):
    # Calculate the prediction with current regression coefficients.
    # We compute costs just for monitoring
    vector = taxilinesCorrected.map(get_vector)
    vector_sum = vector.reduce(lambda x,y:(x[0]+y[0],x[1]+y[1],x[2]+y[2],x[3]+y[3]))

    # calculate cost: cost= (1/n) * sum (( y - y_prediction)**2)
    cost = (1/2*n) * vector_sum[3]
    # calculate gradients.

    m_gradient_vector = (-1.0/n) * vector_sum[2]
    b_gradient = (-1.0/n) * vector_sum[0]

# update the weights - Regression Coefficients
    parameter_current[1:5] = parameter_current[1:5] - learningRate * m_gradient_vector
    parameter_current[0] = parameter_current[0] - learningRate * b_gradient

# Stop if the cost is not descreasing

    if(cost<oldCost):
        learningRate=learningRate*1.05
        oldCost = cost

    if(cost>oldCost):
        learningRate=learningRate*0.5
        oldCost = cost

    print("Iteration No.=", i , " Cost=", cost)
    print(parameter_current)
    print("m = ", parameter_current[1:5], " b=", parameter_current[0])
sc.stop()



