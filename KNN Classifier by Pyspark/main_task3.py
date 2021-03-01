from __future__ import print_function
import sys
import re
import numpy as np
import csv
from numpy import dot
from numpy.linalg import norm
from pyspark import SparkContext
from pyspark.sql import functions as func

f = 20000
sc = SparkContext(appName="Top20KWords")

def buildArray(listOfIndices):
    returnVal = np.zeros(20000)

    for index in listOfIndices:
        returnVal[index] = returnVal[index] + 1

    mysum = np.sum(returnVal)

    returnVal = np.divide(returnVal, mysum)

    return returnVal


def build_zero_one_array(listOfIndices):
    returnVal = np.zeros(20000)

    for index in listOfIndices:
        if returnVal[index] == 0: returnVal[index] = 1

    return returnVal


def stringVector(x):
    returnVal = str(x[0])
    for j in x[1]:
        returnVal += ',' + str(j)
    return returnVal


def cousinSim(x, y):
    normA = np.linalg.norm(x)
    normB = np.linalg.norm(y)
    return np.dot(x, y) / (normA * normB)

if __name__ == "__main__":

   # wikiPagesFile = "WikipediaPagesOneDocPerLine1000LinesSmall.txt"
    #wikiCategoryFile = "wiki-categorylinks-small.csv.bz2"

    # Read two files into RDDs

    wikiCategoryLinks = sc.textFile(sys.argv[1])

    wikiCats = wikiCategoryLinks.map(lambda x: x.split(",")).map(
        lambda x: (x[0].replace('"', ''), x[1].replace('"', '')))
    # Now the wikipages
    wikiPages = sc.textFile(sys.argv[2])

    #df = spark.read.csv(wikiPagesFile)
    # We need this count later ...
    numberOfDocs = wikiPages.count()

    # Each entry in validLines will be a line from the text file
    validLines = wikiPages.filter(lambda x: 'id' in x and 'url=' in x)
    # Now, we transform it into a set of (docID, text) pairs
    keyAndText = validLines.map(lambda x: (x[x.index('id="') + 4: x.index('" url=')], x[x.index('">') + 2:][:-6]))

    # print(keyAndText.take(1))

    # Now, we split the text in each (docID, text) pair into a list of words
    # After this step, we have a data set with
    # (docID, ["word1", "word2", "word3", ...])
    # We use a regular expression here to make
    # sure that the program does not break down on some of the documents

    regex = re.compile('[^a-zA-Z]')

    # remove all non letter characters
    keyAndListOfWords = keyAndText.map(lambda x: (str(x[0]), regex.sub(' ', x[1]).lower().split()))
    # better solution here is to use NLTK tokenizer

    # Now get the top 20,000 words... first change (docID, ["word1", "word2", "word3", ...])
    # to ("word1", 1) ("word2", 1)...
    allWords = keyAndListOfWords.flatMap(lambda x: ((i, 1) for i in x[1]))

    # Now, count all of the words, giving us ("word1", 1433), ("word2", 3423423), etc.
    allCounts = allWords.reduceByKey(lambda x, y: x + y)
    # Get the top 20,000 words in a local array in a sorted format based on frequency
    # If you want to run it on your laptio, it may a longer time for top 20k words.
    topWords = allCounts.top(20000, lambda x: x[1])

    #
    print("Top Words in Corpus:", allCounts.top(10, key=lambda x: x[1]))


    # We'll create a RDD that has a set of (word, dictNum) pairs
    # start by creating an RDD that has the number 0 through 20000
    # 20000 is the number of words that will be in our dictionary
    topWordsK = sc.parallelize(range(20000))

    # Now, we transform (0), (1), (2), ... to ("MostCommonWord", 1)
    # ("NextMostCommon", 2), ...
    # the number will be the spot in the dictionary used to tell us
    # where the word is located
    dictionary = topWordsK.map(lambda x: (topWords[x][0], x))
    print("Word Postions in our Feature Matrix. Last 20 words in 20k positions: ", dictionary.top(20, lambda x: x[1]))


    ################### TASK 2  ##################

    # Next, we get a RDD that has, for each (docID, ["word1", "word2", "word3", ...]),
    # ("word1", docID), ("word2", docId), ...

    allWordsWithDocID = keyAndListOfWords.flatMap(lambda x: ((j, x[0]) for j in x[1]))

    # Now join and link them, to get a set of ("word1", (dictionaryPos, docID)) pairs
    allDictionaryWords = dictionary.join(allWordsWithDocID)

    # Now, we drop the actual word itself to get a set of (docID, dictionaryPos) pairs
    justDocAndPos = allDictionaryWords.map(lambda x: (x[1][1], x[1][0]))

    # Now get a set of (docID, [dictionaryPos1, dictionaryPos2, dictionaryPos3...]) pairs
    allDictionaryWordsInEachDoc = justDocAndPos.groupByKey()

    # The following line this gets us a set of
    # (docID,  [dictionaryPos1, dictionaryPos2, dictionaryPos3...]) pairs
    # and converts the dictionary positions to a bag-of-words numpy array...
    allDocsAsNumpyArrays = allDictionaryWordsInEachDoc.map(lambda x: (x[0], buildArray(x[1])))


    # Now, create a version of allDocsAsNumpyArrays where, in the array,
    # every entry is either zero or one.
    # A zero means that the word does not occur,
    # and a one means that it does.

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

    # use the buildArray function to build the feature array


    print(allDocsAsNumpyArrays.take(3))
    print(allDocsAsNumpyArraysTFidf.take(2))
    featuresRDD = wikiCats.join(allDocsAsNumpyArraysTFidf).map(lambda x: (x[1][0], x[1][1]))
# Cache this important data because we need to run kNN on this data set.
    featuresRDD.cache()


    # Finally, we have a function that returns the prediction for the label of a string, using a kNN algorithm
    def getPrediction(textInput, k):
        # Create an RDD out of the textIput
        myDoc = sc.parallelize(('', textInput))

        # Flat map the text to (word, 1) pair for each word in the doc
        wordsInThatDoc = myDoc.flatMap(lambda x: ((j, 1) for j in regex.sub(' ', x).lower().split()))

        # This will give us a set of (word, (dictionaryPos, 1)) pairs
        allDictionaryWordsInThatDoc = dictionary.join(wordsInThatDoc).map(lambda x: (x[1][1], x[1][0])).groupByKey()

        # Get tf array for the input string
        myArray = buildArray(allDictionaryWordsInThatDoc.top(1)[0][1])

        # Get the tf * idf array for the input string
        myArray = np.multiply(myArray, idfArray)

        # Get the distance from the input text string to all database documents, using cosine similarity (np.dot() )
        distances = featuresRDD.map(lambda x: (x[0], np.dot(x[1], myArray)))
        # distances = allDocsAsNumpyArraysTFidf.map (lambda x : (x[0], cousinSim (x[1],myArray)))
        # get the top k distances
        topK = distances.top(k, lambda x: x[1])

        # and transform the top k distances into a set of (docID, 1) pairs
        docIDRepresented = sc.parallelize(topK).map(lambda x: (x[0], 1))

        # now, for each docID, get the count of the number of times this document ID appeared in the top k
        numTimes = docIDRepresented.reduceByKey(lambda x, y: x + y)

        # Return the top 1 of them.
        # Ask yourself: Why we are using twice top() operation here?
        return numTimes.top(k, lambda x: x[1])


    e=getPrediction('Sport Basketball Volleyball Soccer', 10)
    print(e)
    f=getPrediction('What is the capital city of Australia?', 10)
    print(f)
    g=getPrediction('How many goals Vancouver score last year?', 10)
    print(g)
    sc.stop()
