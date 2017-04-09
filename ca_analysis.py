import numpy as np
import pandas as pd
import cleaning_utils
import mca
import time
import matplotlib.pyplot as plt



data = pd.read_csv('data/train.csv')

###########################
data = data.iloc[0:5000]
###########################


time1 = time.time()

q1 = data[['qid1', 'question1']].drop_duplicates()
q2 = data[['qid2', 'question2']].drop_duplicates()

q1.columns = ['qid', 'question']
q2.columns = ['qid', 'question']

q = pd.concat([q1, q2], ignore_index=True).drop_duplicates()
q.index = q.qid


# Clean questions
q.question = cleaning_utils.clean_data(q.question)
q = q.dropna()


# Create array of words
qw = q.copy()
qw.question = cleaning_utils.removeStopWords(qw.question)
qw = qw.dropna()
qw = cleaning_utils.removeNullWordList(qw)
print " - Questions cleaned", ((time.time()-time1)/60)


time1 = time.time()
qwi, l_words = cleaning_utils.wordsToIndexes(qw)
print " - Words indexed", ((time.time()-time1)/60)


time1 = time.time()
fm = cleaning_utils.indexesToOneHot(qwi, l_words)
print " - One hot created", ((time.time()-time1)/60)


data = None
q1 = None
q2 = None
q = None
qw = None
qwi = None

time1 = time.time()
mca_result = mca.mca(fm, ncols=1000)
print " - MCA finished", ((time.time()-time1)/60)
plt.plot(mca_result.L)

#fm = None
#mca_result = None



