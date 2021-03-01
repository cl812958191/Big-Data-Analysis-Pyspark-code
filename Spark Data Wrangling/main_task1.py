from __future__ import print_function
import sys
from operator import add
from pyspark import SparkContext
sc = SparkContext(appName="ActiveDriver")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: wordcount <file> <output> ", file=sys.stderr)
        exit(-1)
        

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
                    return p
    
    texilinesCorrected = taxilines.filter(correctRows)
    result=texilinesCorrected.map(lambda X: (X[0],1)).reduceByKey(add)
    a=result.top(10, key=lambda X: X[1])
    rdd3=sc.parallelize(a).coalesce(1)
    rdd3.collect()
    rdd3.saveAsTextFile(sys.argv[2])
    sc.stop()

   
