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


def freqArray (listOfIndices, numberofwords):
    returnVal = np.zeros (20000)
    for index in listOfIndices:
        returnVal[index] = returnVal[index] + 1
    returnVal = np.divide(returnVal, numberofwords)
    return returnVal


if __name__ == "__main__":


    sc = SparkContext(appName="LogisticRegressionTask1")

    d_corpus = sc.textFile(sys.argv[1])

    numberOfDocs =d_corpus.count()

    d_keyAndText = d_corpus.map(lambda x : (x[x.index('id="') + 4 : x.index('" url=')], x[x.index('">') + 2:][:-6]))
    regex = re.compile('[^a-zA-Z]')
    d_keyAndListOfWords = d_keyAndText.map(lambda x : (str(x[0]), regex.sub(' ', x[1]).lower().split()))


    #Now get the top 20,000 words... first change (docID, ["word1", "word2", "word3", ...])
    # to ("word1", 1) ("word2", 1)...
    allWords = d_keyAndListOfWords.flatMap(lambda x: ((i,1) for i in x[1]))


    # Now, count all of the words, giving us ("word1", 1433), ("word2", 3423423), etc.
    allCounts = allWords.reduceByKey(lambda x,y: x+y)

    # Get the top 20,000 words in a local array in a sorted format based on frequency
    # If you want to run it on your laptio, it may a longer time for top 20k words.
    topWords = allCounts.top(20000, lambda x : x[1])
    # We'll create a RDD that has a set of (word, dictNum) pairs
    # start by creating an RDD that has the number 0 through 20000
    # 20000 is the number of words that will be in our dictionary

    topWordsK = sc.parallelize(range(20000))

    # Now, we transform (0), (1), (2), ... to ("MostCommonWord", 0)
    # ("NextMostCommon", 1), ...
    # the number will be the spot in the dictionary used to tell us
    # where the word is located
    dictionary = topWordsK.map (lambda x : (topWords[x][0], x))

    # Next, we get a RDD that has, for each (docID, ["word1", "word2", "word3", ...]),
    # ("word1", docID), ("word2", docId), ...

    allWordsWithDocID = d_keyAndListOfWords.flatMap(lambda x: ((j, x[0]) for j in x[1]))

    # Now join and link them, to get a set of ("word1", (dictionaryPos, docID)) pairs
    allDictionaryWords = dictionary.join (allWordsWithDocID)


    # Now, we drop the actual word itself to get a set of (docID, dictionaryPos) pairs
    justDocAndPos = allDictionaryWords.map (lambda x: (x[1][1], x[1][0]))


    # Now get a set of (docID, [dictionaryPos1, dictionaryPos2, dictionaryPos3...]) pairs
    allDictionaryWordsInEachDoc = justDocAndPos.groupByKey()


    # The following line this gets us a set of
    # (docID,  [dictionaryPos1, dictionaryPos2, dictionaryPos3...]) pairs
    # and converts the dictionary positions to a bag-of-words numpy array...
    allDocsAsNumpyArrays = allDictionaryWordsInEachDoc.map(lambda x: (x[0], freqArray(x[1],len(x[1]))))


    zeroOrOne = allDocsAsNumpyArrays.map(lambda x: (x[0], np.clip(np.multiply(x[1], 9e9), 0, 1)))
    # Now, add up all of those arrays into a single array, where the
    # i^th entry tells us how many
    # individual documents the i^th word in the dictionary appeared in
    dfArray = zeroOrOne.reduce(lambda x1, x2: ("", np.add(x1[1], x2[1])))[1]

    # Create an array of 20,000 entries, each entry with the value numberOfDocs (number of docs)
    multiplier = np.full(20000, numberOfDocs)

    # Get the version of dfArray where the i^th entry is the inverse-document frequency for the
    # i^th word in the corpus
    idfArray = np.log(np.divide(np.full(20000, numberOfDocs), dfArray))

    # Finally, convert all of the tf vectors in allDocsAsNumpyArrays to tf * idf vectors
    allDocsAsNumpyArraysTFidf = allDocsAsNumpyArrays.map(lambda x: (x[0], np.multiply(x[1], idfArray)))

    print(allDocsAsNumpyArraysTFidf.take(3))


    # test set
    test_corpus = sc.textFile(sys.argv[2])

    numberOfDocs =test_corpus.count()

    test_keyAndText = test_corpus.map(lambda x : (x[x.index('id="') + 4 : x.index('" url=')], x[x.index('">') + 2:][:-6]))
    regex = re.compile('[^a-zA-Z]')
    test_keyAndListOfWords = test_keyAndText.map(lambda x : (str(x[0]), regex.sub(' ', x[1]).lower().split()))

    t_allWordsWithDocID = test_keyAndListOfWords.flatMap(lambda x: ((j, x[0]) for j in x[1]))

    t_allDictionaryWords = dictionary.join (t_allWordsWithDocID)

    t_justDocAndPos = t_allDictionaryWords.map (lambda x: (x[1][1], x[1][0]))

    t_allDictionaryWordsInEachDoc = t_justDocAndPos.groupByKey()


    t_allDocsAsNumpyArrays = t_allDictionaryWordsInEachDoc.map(lambda x: (x[0], freqArray(x[1],len(x[1]))))

    t_zeroOrOne = t_allDocsAsNumpyArrays.map(lambda x: (x[0], np.clip (np.multiply (x[1], 9e9), 0, 1)))
    t_dfArray = t_zeroOrOne.reduce(lambda x1, x2: ("", np.add(x1[1], x2[1])))[1]
    t_idfArray = np.log(np.divide(np.full(20000, numberOfDocs), t_dfArray))
    t_allDocsAsNumpyArraysTFidf = t_allDocsAsNumpyArrays.map(lambda x: (x[0], np.multiply(x[1], t_idfArray)))

    print(t_allDocsAsNumpyArraysTFidf.take(3))

    application = dictionary.filter(lambda x: x[0] == 'applicant').collect()[0][1]
    and_ = dictionary.filter(lambda x: x[0] == 'and').collect()[0][1]
    attack = dictionary.filter(lambda x: x[0] == 'attack').collect()[0][1]
    protein = dictionary.filter(lambda x: x[0] == 'protein').collect()[0][1]
    court = dictionary.filter(lambda x: x[0] == 'court').collect()[0][1]

    zero_or_one1 = zeroOrOne.map(lambda x: (1 if 'AU' in x[0] else 0, x[1]))
    zero_or_one2 = zero_or_one1.filter(lambda x: x[0] == 1)
    sum1 = zero_or_one2.reduce(lambda x1, x2: ('', np.add(x1[1], x2[1])))[1]
    zero_or_one3 = zero_or_one1.filter(lambda x: x[0] == 0)
    sum2 = zero_or_one3.reduce(lambda x1, x2: ('', np.add(x1[1], x2[1])))[1]

    print('application in AU:', sum1[application] / zero_or_one2.count())
    print('and in AU:', sum1[and_] / zero_or_one2.count())
    print('attack in AU:', sum1[attack] / zero_or_one2.count())
    print('protein in AU:', sum1[protein] / zero_or_one2.count())
    print('court in AU:', sum1[court] / zero_or_one2.count())

    print('application in Wiki:', sum2[application] / zero_or_one3.count())
    print('and in WIki:', sum2[and_] / zero_or_one3.count())
    print('attack in Wiki:', sum2[attack] / zero_or_one3.count())
    print('protein in Wiki:', sum2[protein] / zero_or_one3.count())
    print('court in Wiki:', sum2[court] / zero_or_one3.count())
    sc.stop()