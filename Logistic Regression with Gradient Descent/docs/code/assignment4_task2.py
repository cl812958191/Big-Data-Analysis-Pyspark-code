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
        return 0


def get_vecor_regularization(x):
	return (np.multiply(x[0], x[1]), np.multiply(x[1], (np.e ** float(x[2]) / (1 + np.e ** float(x[2])))),
			x[0] * x[2], np.log(np.e ** float(x[2]) + 1))


if __name__ == "__main__":

	sc = SparkContext(appName="LogisticRegressionTask2")
	d_corpus = sc.textFile(sys.argv[1])

	numberOfDocs = d_corpus.count()

	d_keyAndText = d_corpus.map(lambda x: (x[x.index('id="') + 4: x.index('" url=')], x[x.index('">') + 2:][:-6]))
	regex = re.compile('[^a-zA-Z]')
	d_keyAndListOfWords = d_keyAndText.map(lambda x: (str(x[0]), regex.sub(' ', x[1]).lower().split()))

	allWords = d_keyAndListOfWords.flatMap(lambda x: ((i, 1) for i in x[1]))

	allCounts = allWords.reduceByKey(lambda x, y: x + y)

	topWords = allCounts.top(Dimention, lambda x: x[1])

	topWordsK = sc.parallelize(range(Dimention))
	dictionary = topWordsK.map(lambda x: (topWords[x][0], x))

	allWordsWithDocID = d_keyAndListOfWords.flatMap(lambda x: ((j, x[0]) for j in x[1]))

	allDictionaryWords = dictionary.join(allWordsWithDocID)

	justDocAndPos = allDictionaryWords.map(lambda x: (x[1][1], x[1][0]))

	allDictionaryWordsInEachDoc = justDocAndPos.groupByKey()

	allDocsAsNumpyArrays = allDictionaryWordsInEachDoc.map(lambda x: (x[0], freqArray(x[1], len(x[1]))))
	zeroOrOne = allDocsAsNumpyArrays.map(lambda x: (x[0], np.clip(np.multiply(x[1], 9e9), 0, 1)))
	dfArray = zeroOrOne.reduce(lambda x1, x2: ("", np.add(x1[1], x2[1])))[1]
	multiplier = np.full(Dimention, numberOfDocs)
	idfArray = np.log(np.divide(np.full(Dimention, numberOfDocs), dfArray))
	# X_i
	allDocsAsNumpyArraysTFidf = allDocsAsNumpyArrays.map(lambda x: (x[0], np.multiply(x[1], idfArray)))

	# With Regularization
	R_current2 = np.random.normal(0, 0.1, Dimention)
	learningRate = 0.1
	num_iteration = 9999999
	oldCost = 0

	# I have tested many lambda, lambda=0.00001 can get a good result
	lamda = 0.00001

	allDocsAsNumpyArraysTFidf.cache()

	for i in range(num_iteration):

		Y_X_Theta = allDocsAsNumpyArraysTFidf.map(lambda x: ((get_label(x), x[1], np.dot(x[1], R_current2))))
		Gradient = Y_X_Theta.map(get_vecor_regularization)
		GradientSum = Gradient.reduce(lambda x, y: ((x[0] + y[0]), (x[1] + y[1]), (x[2] + y[2]), (x[3] + y[3])))

		# calculate cost:
		cost = GradientSum[3] - GradientSum[2] + lamda * (np.square(R_current2).sum())

		oldp = np.sqrt(np.square(R_current2).sum())

		R_current2 = R_current2 - learningRate *( (-GradientSum[0] + GradientSum[1])+2*lamda*R_current2)

		p = np.sqrt(np.square(R_current2).sum())

		# Stop if the cost is not descreasing
		if abs(oldp - p) <= 0.1:
			break

		if (cost <= oldCost):
			learningRate = learningRate * 1.05
			oldCost = cost

		if (cost > oldCost):
			learningRate = learningRate * 0.5
			oldCost = cost

	print("Iteration No.=", i, " Cost=", cost)
	print('R current:', R_current2)
	Index = R_current2.argsort()[::-1][:5]
	Top5Words = []
	for i in range(5):
		Top5Words.append(dictionary.filter(lambda x: (x[1] == Index[i])).collect())
	print('Top5Words',Top5Words)
	sc.stop()