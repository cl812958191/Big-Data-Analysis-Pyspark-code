from __future__ import print_function
import sys

from operator import add
from re import sub, search
import numpy as np
from numpy.random.mtrand import dirichlet, multinomial
from string import punctuation
import random
from pyspark import SparkConf, SparkContext
from time import time
f = 20000


if __name__ == "__main__":
    
    
    sc = SparkContext(appName="A6T2-3")

    lines = sc.textFile(sys.argv[1])

    
   
    stripped = lines.map(lambda x: sub("<[^>]+>", "", x))

    # count most frequent words (top 20000)
    def correctWord(p):
        if (len(p) > 2):
            if search("([A-Za-z])\w+", p) is not None:
                return p


    counts = lines.map(lambda x : x[x.find(">") : x.find("</doc>")]) .flatMap(lambda x: x.split()).map(lambda x : x.lower().strip(".,<>()-[]:;?!")) .filter(lambda x : len(x) > 1) .map(lambda x: (x, 1)).reduceByKey(add)


    sortedwords = counts.takeOrdered(f, key = lambda x: -x[1])

    top20000 = []
    for pair in sortedwords:
        top20000.append(pair[0])

    # A function ti generate the wourd count. 
    def countWords(d):
        try:
            header = search('(<[^>]+>)', d).group(1)
        except AttributeError:
            header = ''
        d = d[d.find(">") : d.find("</doc>")]
        words = d.split(' ')
        numwords = {}
        count = 0
        for w in words:
            if search("([A-Za-z])\w+", w) is not None:
                w = w.lower().strip(punctuation)
                if (len(w) > 2) and w in top20000:
                    count += 1
                    idx = top20000.index(w)
                    if idx in numwords:
                        numwords[idx] += 1
                    else:
                        numwords[idx] = 1
        return (header, numwords, count)

    def map_to_array(mapping):
        count_lst = [0] * f
        i = 0
        while i < f:
            if i in mapping:
                count_lst[i] = mapping[i]
            i+= 1
        return np.array(count_lst)

    # calculate term frequency vector for each document
    result = lines.map(countWords)
    result.cache()

    alpha = [0.1] * 20
    beta = np.array([0.1] * f)

    pi = dirichlet(alpha).tolist()
    mu = np.array([dirichlet(beta) for j in range(20)])
    log_mu = np.log(mu)
    header = result.map(lambda x: x[0]).collect()
    n = result.count()
    l = result.map(lambda x: x[2]).collect()
    x = result.map(lambda x: x[1]).map(map_to_array).cache()

    def getProbs(checkParams, log_allMus, x, log_pi):
        if checkParams == True:
    #            if x.shape [0] != log_allMus.shape [1]:
    #                    raise Exception ('Number of words in doc does not match')
            if log_pi.shape[0] != log_allMus.shape [0]:
                    print (log_pi.shape[0])
                    raise Exception ('Number of document classes does not match')
            #if not (0.999 <= np.sum (log_pi) <= 1.001):
            #        raise Exception ('Pi is not a proper probability vector')
            for i in range(log_allMus.shape [0]):
                    if not (0.999 <= np.sum (np.exp (log_allMus[i])) <= 1.001):
                            raise Exception ('log_allMus[' + str(i) + '] is not a proper probability vector')
        allProbs = np.copy(log_pi)
        for i in range(log_allMus.shape [0]):
            product = np.multiply (x, log_allMus[i])
            allProbs[i] += np.sum(product)
        biggestProb = np.amax(allProbs)
        allProbs -= biggestProb
        allProbs = np.exp(allProbs)
        return allProbs / np.sum (allProbs)

    def addup(map1, map2):
        """
        Adds up the values of keys that appear in at least one of the mappings
        """
        map1 = dict(map1) #make a copy so the original dict isn't mutated
        for key in map2:
            if key in map1:
                map1[key] += map2[key]
            else:
                map1[key] = map2[key]
        return map1

    def assignCategory(x, i, c):
        if x[1] == i:
            return c
        else:
            return x[0]
    

    for num_iter in range(200):
        start = time()

        # update c
        logPi = np.log(pi)
        probs = x.map(lambda x_i : getProbs(False, log_mu, x_i, logPi)).collect()
        # print(len(probs)) # 19997
        # Now we need to asign and find out to which category goes each document

        c_local = [np.nonzero(multinomial(1, prob))[0][0] for prob in probs]        
        # print(len(c_local)) # 19997

        c = x.zipWithIndex().map(lambda tup : c_local[tup[1]])
        #make it eager  
        count = c.map(lambda cat : (cat, 1)).reduceByKey(lambda x,y:x+y).sortByKey(ascending = True).collectAsMap()

        new_alpha = [0] * 20
        for i in range(20):
            if i in count:
                new_alpha[i] = alpha[i]+count[i]
            else:
                new_alpha[i] = alpha[i]

        x_c = x.zip(c).cache() 
        empty = sc.parallelize([np.array([0]*f)])

        for j in range(20):
            count = x_c.filter(lambda term : term[1] == j).map(lambda term : term[0]).union(empty).reduce(add)

            log_mu[j] = np.log(dirichlet(np.add(beta,count)))

        stop = time()
        time1 = stop - start
        print('%d iteration takes %f s'%(num_iter,time1))

    tosave = []
    for mu in log_mu:
        mylist = zip(top20000, mu.tolist())
        mylist1 = sorted(mylist, key=(lambda x : x[1]), reverse=True)
        tosave.append(map((lambda y: y[0]), mylist1)[:100])

    sc.parallelize(tosave, 1).saveAsTextFile(sys.argv[2])

    finalresult = sc.parallelize(zip(c.collect(), header)).sortByKey()
    finalresult.saveAsTextFile(sys.argv[3])

    sc.stop()


   


    
    
    







