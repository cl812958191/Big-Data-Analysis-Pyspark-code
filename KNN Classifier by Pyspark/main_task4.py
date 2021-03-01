import sys
import re
import numpy as np
from pyspark import SparkContext
import operator
from collections import Counter
from pyspark.sql import SparkSession
from numpy.linalg import norm
from numpy import dot
from numpy.linalg import norm
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.sql import SQLContext
from pyspark.sql import functions as F
from pyspark.sql.functions import collect_list
from pyspark.ml.linalg import SparseVector, DenseVector
from pyspark.ml.linalg import Vectors, VectorUDT
f = 20000

sc = SparkContext(appName="data-frame")
spark = SparkSession.builder.appName("data-frame").getOrCreate()
# Read two files into DataFrame

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

    # wikiPagesFile="WikipediaPagesOneDocPerLine1000LinesSmall.txt"
    # wikiCategoryFile="wiki-categorylinks-small.csv.bz2"

    # Read two files into DataFrame
    df = spark.read.text(sys.argv[1])
    # Now the wikipages
    wikiCategoryLinks = spark.read.csv(sys.argv[2])

    validLines = df.filter(df.value.contains('id') & df.value.contains('url='))
    # Assumption: Each document is stored in one line of the text file
    # We need this count later ...
    numberOfDocs = df.count()


    # Define two udf to get the id and text for the new dataframe

    get_id_udf = udf(lambda x: x[x.index('id="') + 4: x.index('" url=')])
    get_text_udf = udf(lambda x: x[x.index('">') + 2:][:-6])

    validLinesCopy = validLines

    # Now, we transform it into a set of (docID, text) pairs
    validLines = validLines.withColumn("id", get_id_udf("value"))
    keyAndText = validLines.withColumn("text", get_text_udf("value")).drop('value')

    def tolist(x):
        a = []
        for i in x:
            for j in i:
                a.append(j)
        return a


    tolist_udf = udf(lambda x: tolist(x), ArrayType(StringType(), containsNull=False))

    regex = re.compile('[^a-zA-Z]')
    split_udf = udf(lambda x: regex.sub(' ', x).lower().split(), ArrayType(StringType(), containsNull=False))
    keyAndListOfWords = keyAndText.withColumn('word', split_udf('text')).drop('text')
    keyAndListOfWordsDropID = keyAndListOfWords.drop('id')

    keyAndListOfWordsCount = keyAndListOfWords.withColumn('Index', lit(1))

    keyAndListOfWordsCount = keyAndListOfWordsCount.groupby('Index').agg(collect_list('word').alias("word"))
    allWords = keyAndListOfWordsCount.withColumn('WordsList', tolist_udf('word')).drop('word')
    row_list = allWords.select('WordsList').collect()


    a = allWords.select('WordsList').collect()
    wordString = a[0][0]
    counts = Counter(wordString)
    print("Top Words in Corpus:", counts.most_common(10))

    dfCorpus = spark.createDataFrame(counts.items(), ['words', 'counts'])
    dfSqCorpus = dfCorpus.orderBy(desc("counts")).limit(20000)

    # add the index
    index_list = [x for x in range(20000)]
    idx = 0


    def set_index(x):
        global idx
        if x is not None:
            idx += 1
            return index_list[idx - 1]


    index = udf(set_index, IntegerType())
    dfSqCorpusIndex = dfSqCorpus.select(col("*"), index("words").alias("index")).drop('counts')

    dictionary = dfSqCorpusIndex.orderBy("index", ascending=False)
    dictionary.show()

    # get the ID from the former copyed data frame
    get_TextWithID_udf = udf(lambda x: [x[x.index('id="') + 4: x.index('" url=')], x[x.index('">') + 2:][:-6]],
                             ArrayType(StringType(), containsNull=False))

    IDWord = validLinesCopy.withColumn("id", get_TextWithID_udf("value")).drop('value')

    a = allWords.select('WordsList').collect()[0][0]
    counts = Counter(a)

    dic1 = dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))  #
    dic2 = {}
    num = 0
    for i in dic1.keys():
        dic2[i] = num
        num = num + 1
        if num == f:
            break


    def IDList(x):
        id = x[0]
        list1 = [id]
        t = []
        regex = re.compile('[^a-zA-Z]')
        a = regex.sub(' ', x[1])
        for i in a.lower().split():
            if i in dic2.keys():
                t.append(dic2[i])
        list1.append(t)
        return list1


    id_list_udf = udf(lambda x: IDList(x), ArrayType(StringType(), containsNull=False))
    IDDicList = IDWord.withColumn("id_list", id_list_udf("id")).drop('id')
    #IDDicList.show()

    def array(x):
        id = x[0]
        list_all = [id]
        list_all.append(buildArray(list(x[1])))
        return list_all


    array_udf = udf(lambda x: array(x))

    allDocsAsNumpyArrays = IDDicList.withColumn("array_list", array_udf("id_list")).drop('id_list')


    def zero_one(x):
        t = []
        for i in x[1]:
            if i > 0:
                t.append(1)
            else:
                t.append(0)
        return t


    add_ID_udf = udf(lambda x: x[0])
    add_list_udf = udf(lambda x: x[1])
    zero_one_udf = udf(lambda x: zero_one(x), ArrayType(IntegerType(), containsNull=False))
    list0 = allDocsAsNumpyArrays.withColumn("DocID", add_ID_udf("array_list"))
    list1 = list0.withColumn('list',add_list_udf('array_list')).drop('array_list')
    zeroOrOne = list0.withColumn("zero_one_list", zero_one_udf("array_list")).drop('array_list')
    #zeroOrOne.show()

    num1 = len(zeroOrOne.select('zero_one_list').first()[0])

    resultDF = \
    zeroOrOne.agg(F.array(*[F.sum(F.col("zero_one_list")[i]) for i in range(num1)]).alias("sum")).select('sum').collect()[
        0][0]

    multiplier = np.full(f, numberOfDocs)
    idfArray = np.log(np.divide(np.full(f, numberOfDocs), resultDF))

    TFidf = udf(lambda x: np.multiply(x, idfArray).tolist(), ArrayType(FloatType(), containsNull=False))

    allDocsAsNumpyArraysTFidf = list1.withColumn("TFidf", TFidf("list")).drop('list')

    featuresRDD = wikiCategoryLinks.join(allDocsAsNumpyArraysTFidf,
                                         wikiCategoryLinks._c0 == allDocsAsNumpyArraysTFidf.DocID).drop('_c0').drop(
        'DocID')

    #allDocsAsNumpyArrays.take(3)

    #allDocsAsNumpyArraysTFidf.take(2)


    def getPrediction(textInput, k):
        # Create an RDD out of the textIput

        input_text = spark.sparkContext.parallelize([textInput])
        input_text.take(1)
        myDoc = input_text.map(lambda x: (x,)).toDF()

        split1_udf = udf(lambda x: regex.sub(' ', x).lower().split(), ArrayType(StringType(), containsNull=False))
        allWordssplit = myDoc.withColumn('words', split1_udf('_1')).drop('_1')

        def get_id_text_list_single1(x):
            asd = []
            for i in x:
                if i in dic2.keys():
                    asd.append(int(dic2[i]))
            return asd

        get_id_text_list_single_udf = udf(lambda x: get_id_text_list_single1(x),
                                      ArrayType(IntegerType(), containsNull=False))
        allDictionaryWordsInThatDoc = allWordssplit.withColumn('words_indic', get_id_text_list_single_udf('words')).drop(
            'words')

        myArray = buildArray(allDictionaryWordsInThatDoc.select("words_indic").collect()[0][0])

        myArray = np.multiply(np.array(myArray), idfArray)

        def get_dot1(x):
            asd = np.dot(np.array(x), myArray)
            return asd

        get_dot_udf = udf(lambda x: get_dot1(x).tolist(), FloatType())

        distances = featuresRDD.withColumn('rank', get_dot_udf('TFidf')).drop('TFidf')

        topK = distances.orderBy(desc("rank"), '_c1').limit(k)
        add_one_udf = udf(lambda x: 1, IntegerType())

        docIDRepresented = topK.withColumn('times', add_one_udf('rank')).drop('rank')

        docIDRepresented1 = docIDRepresented.groupBy('_c1').agg(F.sum('times').alias('sum'))

        numTimes = docIDRepresented1.orderBy(desc("sum")).limit(k)
        asd= []
        for item in numTimes.collect():
            asd.append((item[0], item[1]))
        return asd


    print(getPrediction('Sport Basketball Volleyball Soccer', 10))

    print(getPrediction('What is the capital city of Australia?', 10))

    print(getPrediction('How many goals Vancouver score last year?', 10))

