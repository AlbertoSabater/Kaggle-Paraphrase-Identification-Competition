import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords

import dask.dataframe as dd
import dask
from sklearn import preprocessing



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


def removeStopWords(questions):
    questions = questions.apply(lambda x: x.split())
    questions = questions.apply(lambda x: [item for item in x if item not in stop])

    return questions


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
    qwidd = dd.from_pandas(qwi, npartitions=8)
    aux = qwidd.question.apply(lambda x: le.transform(x), meta=('x', list)).compute()
    qwi.question = aux
    
    # le.inverse_transform(np.where(fm.iloc[[0]].as_matrix()[0] == 1))
    return qwi, len(words)


# Transform array of indexes to one hot
def indexesToOneHot(qwi, l_words):
    fm = pd.DataFrame(0, index=qwi.index, columns=np.arange(l_words))
    print fm.shape
    for index, row in qwi.iterrows():
        for ind in row.question:
            fm.loc[[index], [ind]] = 1

    return fm


