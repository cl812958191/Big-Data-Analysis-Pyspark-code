from __future__ import print_function
import sys
import re
import numpy as np
from numpy import dot
from numpy.linalg import norm
from operator import add
from pyspark import SparkContext
import pandas as pd


sc = SparkContext(appName="Linear Regression")


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
texilinesCorrected = taxilines.filter(correctRows)

n=texilinesCorrected.count()

x=texilinesCorrected.map(get_variable).reduce(lambda x,y: (float(x[0])+float(y[0]), float(x[1])+float(y[1]),
                                                           float(x[2])+float(y[2]), float(x[3])+float(y[3])))

m=(x[3]*n-x[0]*x[1])/(n*x[2]-x[0]**2)
print(m)

b=(x[2]*x[1]-x[0]*x[3])/(n*x[2]-x[0]**2)
print(b)
sc.stop()




