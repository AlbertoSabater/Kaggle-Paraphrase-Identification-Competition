import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

import dask.dataframe as dd
import dask
from sklearn import preprocessing

import sys



np.random.seed(1337)  # for reproducibility
stop = stopwords.words('english')


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
        if current % 25000 == 0:
            showProgress(current, limit)
            
        for ind in row.question:
            fm.loc[[index], [ind]] = 1
        current += 1

    return fm


def showProgress(current, limit):
    bar_len = 60
    filled_len = int(round(bar_len * current / limit))

    percents = round(100.0 * current / limit, 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...\r' % (bar, percents, '%'))
    sys.stdout.flush()  # As suggested by Rom Ruben