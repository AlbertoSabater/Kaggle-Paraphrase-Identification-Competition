import numpy as np
import pandas as pd
import cleaning_utils
import mca
import time
import matplotlib.pyplot as plt
import pickle
import sys


LOAD_QWI = True


if not LOAD_QWI:
    data = pd.read_csv('data/train.csv')
    
    ###########################
    #data = data.iloc[0:50000]
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
    
    print q.shape
    print qw.shape
    
    
    '''
    import collections
    aux = np.concatenate(np.array(qw.question))
    counter = collections.Counter(aux)
    print(counter.most_common(3))

    unique_words = []
    for key, value in counter.iteritems():
        if value == 1:
            print key, "-", value
            unique_words += [key]
    unique_words = list(set(unique_words))
    
    qw.question = qw.question.apply(lambda x: [item for item in x if item not in unique_words])
    '''

    time1 = time.time()
    qw = cleaning_utils.removeUniqueWords(qw, 1)
    print " - Unique words cleaned", ((time.time()-time1)/60)
    
    print qw.shape
     
    qw = cleaning_utils.removeNullWordList(qw)
    
    print qw.shape
    
    with open('qw_unique.pickle', 'w') as f:  # Python 3: open(..., 'wb')
        pickle.dump(qw, f)
        
        
        
    time1 = time.time()
    qwi, l_words = cleaning_utils.wordsToIndexes(qw)
    print " - Words indexed", ((time.time()-time1)/60)
    
    with open('qwi_unique.pickle', 'w') as f:  # Python 3: open(..., 'wb')
        pickle.dump([qwi, l_words], f)
    
else:
    print "Loading qwi..."
    with open('qwi_unique.pickle', 'r') as f:  # Python 3: open(..., 'rb')
        qwi, l_words = pickle.load(f)
    print "qwi loaded"
    



''' Training PCA by batchs '''

from sklearn.decomposition import IncrementalPCA
time1 = time.time()
ipca = IncrementalPCA(n_components=2000)

steps = np.append(np.arange(0, qwi.shape[0], 10000), qwi.shape[0])

time_total = time.time()

for i in range(len(steps)-1):
    time1 = time.time()
    qwi_aux =  qwi[steps[i]:steps[i+1]]
    fm = cleaning_utils.indexesToOneHot(qwi_aux, l_words)
    qwi_aux = None
    
    ipca.partial_fit(fm)
    
#    print "   - Storing", i
#    with open('fm_' + str(i) + '.pickle', 'w') as f:  # Python 3: open(..., 'wb')
#        pickle.dump(fm, f)
        
    fm = None
    
    print " - " + str(i) + ": One hot created", ((time.time()-time1)/60)

print " - Total One hot created", ((time.time()-time_total)/60)


with open('ipca.pickle', 'w') as f:  # Python 3: open(..., 'wb')
    pickle.dump(ipca, f)
    

# TO_DO: load all fm to transform it or transform by batches
fm_pca = ipca.fit_transform(fm)

with open('fm_pca.pickle', 'w') as f:  # Python 3: open(..., 'wb')
    pickle.dump(fm_pca, f)


#
#time1 = time.time()
#mca_result = mca.mca(fm, ncols=1000)
#print " - MCA finished", ((time.time()-time1)/60)
#plt.plot(mca_result.L)    