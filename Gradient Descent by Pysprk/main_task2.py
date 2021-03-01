from __future__ import print_function
import sys
import re
import numpy as np
from numpy import dot
from numpy.linalg import norm
from operator import add
from pyspark import SparkContext



sc = SparkContext(appName="Gradient Descent")

       

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



taxilinesCorrected = taxilines.filter(correctRows)


b_current = 0.1
m_current = 0.1

learningRate = 0.0001
num_iteration = 100


previous_step_size = 1

n=taxilinesCorrected.count()
oldCost=0
# Start of iterations
# Capital X and Y are numpy Arrays
# These two variables are just for visualtion 



# Let's start with main iterative part of gradient descent algorithm 
def calculate_gradient(p):
    return (float(p[5])*float(p[11]),float(p[5])*(m_current * float(p[5]) + b_current),
           p[11],(m_current * float(p[5]) + b_current))

def calculate_cost(p):
    return ((float(p[11])-(m_current * float(p[5]) + b_current))**2)


taxilinesCorrected.cache()
for i in range(num_iteration):
    gradient=taxilinesCorrected.map(calculate_gradient).reduce(lambda x,y: (float(x[0])+float(y[0]),
                                                                        float(x[1])+float(y[1]),
                                                                        float(x[2])+float(y[2]),
                                                                        float(x[3])+float(y[3])))



    cost_sum=taxilinesCorrected.map(calculate_cost).reduce(lambda x,y: (x+y))

    cost= (1/n) * cost_sum

    m_gradient=(-2.0/n) *(gradient[0]-gradient[1])
    b_gradient=(-2.0/n) *(gradient[2]-gradient[3])

# update the weights - Regression Coefficients
    m_current = m_current - learningRate * m_gradient
    b_current = b_current - learningRate * b_gradient



    if(cost<oldCost):
        learningRate=learningRate*1.05
        oldCost = cost

    if(cost>oldCost):
        learningRate=learningRate*0.5
        oldCost = cost


    print("Iteration No.=", i , " Cost=", cost)
    print("m = ", m_current, " b=", b_current)
sc.stop()






