from __future__ import print_function
import sys
import re
import numpy as np
import csv
from numpy import dot
from numpy.linalg import norm
from pyspark import SparkContext
from pyspark.sql import functions as func

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
   
