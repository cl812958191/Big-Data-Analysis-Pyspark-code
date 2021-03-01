from __future__ import print_function
import sys
import re
import numpy as np

from numpy import dot
from numpy.linalg import norm
from pyspark import SparkContext
from operator import add
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.regression import LabeledPoint
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from time import time

Dimention=20000

def freqArray (listOfIndices, numberofwords):
    returnVal = np.zeros (Dimention)
    for index in listOfIndices:
        returnVal[index] = returnVal[index] + 1
    returnVal = np.divide(returnVal, numberofwords)
    return returnVal

def get_label(x):
    if 'AU' in x[0]:
        return 1
    else:
        return -1
    
def get_vector(x):
    return (np.max([0,(1-x[0]*np.dot(w_current,x[1]))]),0 if (1-x[0]*np.dot(w_current,x[1])) < 0 else (-x[0]*x[1]))

if __name__ == "__main__":
    
        
    sc = SparkContext(appName="SVM")
    # load training set
    start_time = time()

    d_corpus = sc.textFile(sys.argv[1])
    t_d_corpus = sc.textFile(sys.argv[2])


    numberOfDocs =d_corpus.count()

    d_keyAndText = d_corpus.map(lambda x : (x[x.index('id="') + 4 : x.index('" url=')], x[x.index('">') + 2:][:-6]))
    regex = re.compile('[^a-zA-Z]')
    d_keyAndListOfWords = d_keyAndText.map(lambda x : (str(x[0]), regex.sub(' ', x[1]).lower().split()))

    allWords = d_keyAndListOfWords.flatMap(lambda x: ((i,1) for i in x[1]))

    allCounts = allWords.reduceByKey(lambda x,y: x+y)
    allCounts.cache()
    topWords = allCounts.top(Dimention, lambda x : x[1])

    topWordsK = sc.parallelize(range(Dimention))
    dictionary = topWordsK.map (lambda x : (topWords[x][0], x))

    allWordsWithDocID = d_keyAndListOfWords.flatMap(lambda x: ((j, x[0]) for j in x[1]))

    allDictionaryWords = dictionary.join (allWordsWithDocID)


    justDocAndPos = allDictionaryWords.map (lambda x: (x[1][1], x[1][0]))

    allDictionaryWordsInEachDoc = justDocAndPos.groupByKey()

    allDocsAsNumpyArrays = allDictionaryWordsInEachDoc.map(lambda x: (x[0], freqArray(x[1],len(x[1]))))

    allDocsAsNumpyArrays.cache()
    print(allDocsAsNumpyArrays.take(3))
    
        # load test set
    t_numberOfDocs =t_d_corpus.count()

    t_d_keyAndText = t_d_corpus.map(lambda x : (x[x.index('id="') + 4 : x.index('" url=')], x[x.index('">') + 2:][:-6]))
    regex = re.compile('[^a-zA-Z]')
    t_d_keyAndListOfWords = t_d_keyAndText.map(lambda x : (str(x[0]), regex.sub(' ', x[1]).lower().split()))

    t_allWords = t_d_keyAndListOfWords.flatMap(lambda x: ((i,1) for i in x[1]))

    t_allCounts = t_allWords.reduceByKey(lambda x,y: x+y)
    t_allCounts.cache()
    t_topWords = t_allCounts.top(Dimention, lambda x : x[1])

    topWordsK = sc.parallelize(range(Dimention))
    t_dictionary = topWordsK.map (lambda x : (t_topWords[x][0], x))

    t_allWordsWithDocID = t_d_keyAndListOfWords.flatMap(lambda x: ((j, x[0]) for j in x[1]))

    t_allDictionaryWords = t_dictionary.join (t_allWordsWithDocID)


    t_justDocAndPos = t_allDictionaryWords.map (lambda x: (x[1][1], x[1][0]))

    t_allDictionaryWordsInEachDoc = t_justDocAndPos.groupByKey()

    t_allDocsAsNumpyArrays = t_allDictionaryWordsInEachDoc.map(lambda x: (x[0], freqArray(x[1],len(x[1]))))
    t_allDocsAsNumpyArrays.cache()
    print(t_allDocsAsNumpyArrays .take(3))
    load_data_duration=time() - start_time

        # SVM
    w_current=np.random.normal(0,10,Dimention)
    learningRate=0.5
    num_iteration = 300
    oldCost=0
    n=numberOfDocs
    lamda = 100
    
    
    allDocsAsNumpyArrays.cache()
    Y_X_RDD=allDocsAsNumpyArrays.map(lambda x: ((get_label(x), x[1])))
    train_start_time = time()
    for i in range(num_iteration):


        Gradient=Y_X_RDD.map(get_vector)
        GradientSum = Gradient.reduce(lambda x,y: ((x[0] + y[0]),(x[1] + y[1])))

        # calculate cost:
        cost = (1/(2*(n*lamda)))*np.dot(w_current,w_current)+(GradientSum[0]/n)

        oldp = np.sqrt(abs(np.dot(w_current,w_current)))
        temp=w_current

        w_current= w_current-learningRate*(((1/(n*lamda))*w_current)+(GradientSum[1]/n))

        p = np.sqrt(abs(np.dot(temp,w_current)))

        #Stop if the cost is not descreasing 
        if abs(oldp-p) <= 0.00005:
            break

        if(cost<oldCost):
            learningRate=learningRate*1.05
            oldCost = cost

        if(cost>oldCost):
            learningRate=learningRate*0.5
            oldCost = cost


        print("Iteration No.=", i+1 , " Cost=", cost) 
        print('w current:',w_current)
    print('finish')
    train_duration= time()-train_start_time
    test_start_time=time()
    y_ypred = t_allDocsAsNumpyArrays.map(
            lambda x: (1 if 'AU' in x[0] else -1, 1 if np.dot(x[1], w_current) > 0 else -1)).collect()

    y = []
    ypred = []
    counter = 0
    for i in y_ypred:
        y.append(i[0])
        ypred.append(i[1])
    print('f1 score:', f1_score(y, ypred, average='binary'))
    print('confusion matrix:', confusion_matrix(y, ypred))
    test_duration=time()-test_start_time
    total_duration=time() - start_time
    print('load data duration(s):%.4f, train model duration(s):%.4f, test model duration(s):%.4f total duration(s): %.4f'
        %(load_data_duration,train_duration,test_duration,total_duration))

    
   
    
    sc.stop()
