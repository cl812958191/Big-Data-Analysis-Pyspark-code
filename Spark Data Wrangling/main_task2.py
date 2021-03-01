from __future__ import print_function

import sys
from operator import add

from pyspark import SparkContext
sc = SparkContext(appName="BestDriver")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: wordcount <file> <output> ", file=sys.stderr)
        exit(-1)
        
   lines = sc.textFile(sys.argv[1])
   taxilines = lines.map(lambda X: X.split(','))


    def isfloat(value):
        try:
            float(value)
            return True
        except:
            return False

    def correctRows(p):
        if (len(p) == 17):
            if (isfloat(p[5])) and isfloat(p[11]):
                if (float(p[5] != 0) and float(p[1] != 0 and float(p[4]) != 0 and float(p[16]) != 0)):
                    return p


    def average(taxilines):
        dv = taxilines[0]
        time = taxilines[1][0]
        money = taxilines[1][1]
        average = float(money) / (float(time) /60)
        return (dv, average)


    texilinesCorrected = taxilines.filter(correctRows)
    x = texilinesCorrected.map(lambda X: (X[1], (X[4], X[16]))).reduceByKey(lambda x, y: (float(x[0]) + float(y[0]), float(x[1]) + float(y[1])))
    y = x.map(average).reduceByKey(lambda x: x[0])
    result=y.top(10, key=lambda y: y[1])

    k=sc.parallelize(result).coalesce(1)
    k.saveAsTextFile(sys.argv[2])
    sc.stop()





