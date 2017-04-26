import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

import dask.dataframe as dd
import dask
from sklearn import preprocessing
import collections

import sys

import math
from collections import Counter


np.random.seed(1337)  # for reproducibility
stop = stopwords.words('english')
dask.set_options(get=dask.multiprocessing.get)


# Remove quotes on the left and right side
def removeExternalQuotes(s):
    result = re.match(r'^\"(.*)\"$', s)
    if result:
        return s[1:-1]
    else:
        return s
    
    
def clean_data(questions):
    questions = questions.apply(lambda x: str(x))
    questions = questions.apply(lambda x: x.lower())
    #questions = q.question.apply(lambda x: x.rstrip('?"\''))
    questions = questions.apply(lambda x: re.sub('[,\.\(\)?"\']','',x))
    questions = questions.apply(lambda x: x.strip())
    #q.question = q.question.apply(lambda x: removeExternalQuotes(x))
    
    return questions


def cleanMoreStopWords(text):
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"what's", "", text)
    text = re.sub(r"What's", "", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"I'm", "I am", text)
    text = re.sub(r" m ", " am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\0k ", "0000 ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e-mail", "email", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"quikly", "quickly", text)
    text = re.sub(r" usa ", " America ", text)
    text = re.sub(r" USA ", " America ", text)
    text = re.sub(r" u s ", " America ", text)
    text = re.sub(r" uk ", " England ", text)
    text = re.sub(r" UK ", " England ", text)
    text = re.sub(r"india", "India", text)
    text = re.sub(r"china", "China", text)
    text = re.sub(r"chinese", "Chinese", text) 
    text = re.sub(r"imrovement", "improvement", text)
    text = re.sub(r"intially", "initially", text)
    text = re.sub(r"quora", "Quora", text)
    text = re.sub(r" dms ", "direct messages ", text)  
    text = re.sub(r"demonitization", "demonetization", text) 
    text = re.sub(r"actived", "active", text)
    text = re.sub(r"kms", " kilometers ", text)
    text = re.sub(r"KMs", " kilometers ", text)
    text = re.sub(r" cs ", " computer science ", text) 
    text = re.sub(r" upvotes ", " up votes ", text)
    text = re.sub(r" iPhone ", " phone ", text)
    text = re.sub(r"\0rs ", " rs ", text) 
    text = re.sub(r"calender", "calendar", text)
    text = re.sub(r"ios", "operating system", text)
    text = re.sub(r"gps", "GPS", text)
    text = re.sub(r"gst", "GST", text)
    text = re.sub(r"programing", "programming", text)
    text = re.sub(r"bestfriend", "best friend", text)
    text = re.sub(r"dna", "DNA", text)
    text = re.sub(r"III", "3", text) 
    text = re.sub(r"the US", "America", text)
    text = re.sub(r"Astrology", "astrology", text)
    text = re.sub(r"Method", "method", text)
    text = re.sub(r"Find", "find", text) 
    text = re.sub(r"banglore", "Banglore", text)
    text = re.sub(r" J K ", " JK ", text)
    
    return text


def removeStopWordsDist(questions):   
    aux = questions.apply(lambda x: str(x).decode('utf-8'))
    aux = dd.from_pandas(aux, npartitions=16)
    return aux.apply(lambda x: ' '.join(filter(lambda y: y.lower() not in stop,  x.split())).encode('utf-8')).compute()



def removeStopWords(questions):   
    questions = questions.apply(lambda x: cleanMoreStopWords(x))
    questions = questions.apply(lambda x: x.split())
    questions = questions.apply(lambda x: [item for item in x if item not in stop])
    questions = questions.apply(lambda x: stemming(x))

    return questions


def stemming(questions):
     stemmer = SnowballStemmer('english')
     stemmed_words = [stemmer.stem(word) for word in questions]
     return stemmed_words


# questions = array of words
def getWords(questions):
    words = []
    for ans in questions:
        words += ans
        
    return list(set(words))


# Remove questions with lenght == 0
def removeNullWordList(qw):
    return qw[qw.question.apply(lambda x: len(x)) != 0]


def removeUniqueWords(qw, maxRepetition):
    aux = np.concatenate(np.array(qw.question))
    counter = collections.Counter(aux)
    print(counter.most_common(3))

    unique_words = []
    for key, value in counter.iteritems():
        if value <= maxRepetition:
            #print key, "-", value
            unique_words += [key]
    unique_words = list(set(unique_words))
    
    print "Unique words calculated. Tamano", len(unique_words)
    
    qwdd = dd.from_pandas(qw, npartitions=16)
    print "dd creado"
    aux = qwdd.question.apply(lambda x: [item for item in x if item not in unique_words], meta=('x', list)).compute()
    qw.question = aux
    
    return qw



# Transform array of words to array of indexes
def wordsToIndexes(qw):
    words = getWords(qw.question)
    le = preprocessing.LabelEncoder()
    le = le.fit(words)

    dask.set_options(get=dask.multiprocessing.get)
    qwi = qw.copy()
    qwidd = dd.from_pandas(qwi, npartitions=16)
    aux = qwidd.question.apply(lambda x: le.transform(x), meta=('x', list)).compute()
    qwi.question = aux
    
    # le.inverse_transform(np.where(fm.iloc[[0]].as_matrix()[0] == 1))
    return qwi, len(words)


# Transform array of indexes to one hot
def indexesToOneHot(qwi, l_words):
    fm = pd.DataFrame(0, index=qwi.index, columns=np.arange(l_words))
    print fm.shape
    
    limit = float(len(fm.index))
    current = float(1)
    for index, row in qwi.iterrows():
        if current % 5000 == 0:
            showProgress(current, limit)
        
#        i = np.zeros(l_words)
#        i[row.question] = 1
#        fm.loc[[index]] = i
        
        for ind in row.question:
            fm.loc[[index], [ind]] = 1
        current += 1

    return fm


def trainTfIdf():
    return









# Distances: http://web.archive.org/web/20081224234350/http://www.dcs.shef.ac.uk/~sam/stringmetrics.html
def getTextFeaturesPairwiseDistributed(q1, q2):
    #from sklearn.metrics import jaccard_similarity_score
    import Levenshtein
    
    ddq = pd.concat([q1,q2], axis=1)
    ddq.question1 = ddq.question1.apply(lambda x: str(x).decode('utf-8'))
    ddq.question2 = ddq.question2.apply(lambda x: str(x).decode('utf-8'))

    textFeatures = pd.DataFrame(columns =  ['cos_sim', 'lev_dist', 'jaro', 'jaro_winkler', 'ratio'])

    aux = dd.from_pandas(ddq, npartitions=16)
    
    textFeatures.cos_sim = aux.apply(lambda x: cosine_similarity(text_to_vector(x.question1), text_to_vector(x.question2)), axis=1, meta=('x', list)).compute()
    textFeatures.lev_dist = aux.apply(lambda x: Levenshtein.distance(x.question1, x.question2), axis=1, meta=('x', list))
    textFeatures.jaro = aux.apply(lambda x: Levenshtein.jaro(x.question1, x.question2), axis=1, meta=('x', list))
    textFeatures.jaro_winkler = aux.apply(lambda x: Levenshtein.jaro_winkler(x.question1, x.question2), axis=1, meta=('x', list))
    textFeatures.ratio = aux.apply(lambda x: Levenshtein.ratio(x.question1, x.question2), axis=1, meta=('x', list))
      
    return textFeatures


## Distances: http://web.archive.org/web/20081224234350/http://www.dcs.shef.ac.uk/~sam/stringmetrics.html
#def getTextFeaturesPairwise(q1, q2, index):
#    #from sklearn.metrics import jaccard_similarity_score
#    import Levenshtein
#    
#    print len(q1), len(q2), len(index)
#    
#    textFeatures = pd.DataFrame(columns=['cos_sim', 'lev_dist', 'jaro', 'jaro_winkler', 'ratio'])
#    
#    limit = float(len(index))
#    current = float(1)
#        
#    for s1, s2, i in zip(q1,q2,index):
#        
#        if current % 10000 == 0:
#            showProgress(current, limit)
#        
##        print "\n\n", s1, s2, "\n"
#        #jaccard = jaccard_similarity_score(row.question1.split(), row.question2.split(), normalize=True)
#        cos_sim = cosine_similarity(text_to_vector(s1), text_to_vector(s2))
#        
#        dist = Levenshtein.distance(s1, s2)
#        jaro = Levenshtein.jaro(s1, s2)
#        jaro_winkler = Levenshtein.jaro_winkler(s1, s2)
#        ratio = Levenshtein.ratio(s1, s2)
#        
##        print [cos_sim, dist, jaro, jaro_winkler, ratio], s1.index, s2.index
#        textFeatures[i] = [cos_sim, dist, jaro, jaro_winkler, ratio]
# 
#        current += 1
#      
#    return textFeatures


#https://gist.github.com/ahmetalsan/06596e3f2ea3182e185a
def cosine_similarity(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x]**2 for x in vec1.keys()])
    sum2 = sum([vec2[x]**2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator

def text_to_vector(text):
    word = re.compile(r'\w+')
    words = word.findall(text)
    return Counter(words)
 




def showProgress(current, limit):
    bar_len = 60
    filled_len = int(round(bar_len * current / limit))

    percents = round(100.0 * current / limit, 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...\r' % (bar, percents, '%'))
    sys.stdout.flush()  # As suggested by Rom Ruben