import os
import re
import csv
import codecs
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from string import punctuation

#from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
#from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, multiply
#from keras.layers.merge import concatenate
#from keras.models import Model, Sequential
#from keras.layers.normalization import BatchNormalization
#from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.callbacks import ModelCheckpoint, EarlyStopping

import pickle
import cleaning_utils
import models
import time

np.random.seed(1337)  # for reproducibility


# TO-DO: define max words
MAX_NB_WORDS = 600000           # 424706
MAX_SEQUENCE_LENGTH = 30
MAX_SEQUENCE_LENGTH_NOSTOPWORDS = 30

NUM_LSTM = 128

LOAD_ALL = False


if not LOAD_ALL:
    data_train = pd.read_csv('data/train.csv')
    data_test = pd.read_csv('data/test.csv')
    
    
    q1 = data_train['question1']
    q2 = data_train['question2']
    labels = data_train['is_duplicate']
    
    q1_test = data_test['question1']
    q2_test = data_test['question2']
    
    
    
    
    time1 = time.time()
    # Clean questions
    q1 = cleaning_utils.clean_data(q1)
    q2 = cleaning_utils.clean_data(q2)
    q1_test = cleaning_utils.clean_data(q1_test)
    q2_test = cleaning_utils.clean_data(q2_test)
    print "Data cleaned", ((time.time()-time1)/60)
    
    
    
    time1 = time.time()
    textDistances = cleaning_utils.getTextFeaturesPairwiseDistributed(q1,q2)
    textDistances_test = cleaning_utils.getTextFeaturesPairwiseDistributed(q1_test,q2_test)
    print "Distances calculated", ((time.time()-time1)/60)
    
    

    time1 = time.time()
    q1_noStopWords = cleaning_utils.removeStopWordsDist(q1)
    q2_noStopWords = cleaning_utils.removeStopWordsDist(q2)
    q1_test_noStopWords = cleaning_utils.removeStopWordsDist(q1_test)
    q2_test_noStopWords = cleaning_utils.removeStopWordsDist(q2_test)
    print "Stop words removed", ((time.time()-time1)/60)



    time1 = time.time()
    textDistances_noStopWords = cleaning_utils.getTextFeaturesPairwiseDistributed(q1_noStopWords,q2_noStopWords)
    textDistances_test_noStopWords = cleaning_utils.getTextFeaturesPairwiseDistributed(q1_test_noStopWords,q2_test_noStopWords)
    print "Distances noStopWords calculated", ((time.time()-time1)/60)
        
        
    
    time1 = time.time()
    
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(list(q1) + list(q2) + list(q1_test) + list(q2_test))
    
    train_1 = tokenizer.texts_to_sequences(q1)
    train_2 = tokenizer.texts_to_sequences(q2)
    test_1 = tokenizer.texts_to_sequences(q1_test)
    test_2 = tokenizer.texts_to_sequences(q2_test)
    
    train_1 = pad_sequences(train_1, maxlen=MAX_SEQUENCE_LENGTH)
    train_2 = pad_sequences(train_2, maxlen=MAX_SEQUENCE_LENGTH)
    test_1 = pad_sequences(test_1, maxlen=MAX_SEQUENCE_LENGTH)
    test_2 = pad_sequences(test_2, maxlen=MAX_SEQUENCE_LENGTH)
    
    nb_words = min(MAX_NB_WORDS, len(tokenizer.word_index))+1
    print "Tokenizer created", ((time.time()-time1)/60)
    
    
    
    time1 = time.time()
    
    tokenizer_noStopWords = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer_noStopWords.fit_on_texts(list(q1_noStopWords) + list(q2_noStopWords) + list(q1_test_noStopWords) + list(q2_test_noStopWords))
    
    train_1_noStopWords = tokenizer_noStopWords.texts_to_sequences(q1_noStopWords)
    train_2_noStopWords = tokenizer_noStopWords.texts_to_sequences(q2_noStopWords)
    test_1_noStopWords = tokenizer_noStopWords.texts_to_sequences(q1_test_noStopWords)
    test_2_noStopWords = tokenizer_noStopWords.texts_to_sequences(q2_test_noStopWords)
    
    train_1_noStopWords = pad_sequences(train_1_noStopWords, maxlen=MAX_SEQUENCE_LENGTH)
    train_2_noStopWords = pad_sequences(train_2_noStopWords, maxlen=MAX_SEQUENCE_LENGTH)
    test_1_noStopWords = pad_sequences(test_1_noStopWords, maxlen=MAX_SEQUENCE_LENGTH)
    test_2_noStopWords = pad_sequences(test_2_noStopWords, maxlen=MAX_SEQUENCE_LENGTH)
    
    nb_words_noStopWords = min(MAX_NB_WORDS, len(tokenizer_noStopWords.word_index))+1
    print "Tokenizer created NoStopWords", ((time.time()-time1)/60)
    


    time1 = time.time()
    with open('calculated_variables/all_training_variables_yesNoStopWords.pickle', 'w') as f:  # Python 3: open(..., 'wb')
        pickle.dump([train_1, train_2, labels, test_1, test_2, 
                     nb_words, tokenizer, textDistances, textDistances_test, 
                     train_1_noStopWords, train_2_noStopWords, test_1_noStopWords, test_2_noStopWords, 
                     nb_words_noStopWords, tokenizer_noStopWords, textDistances_noStopWords, textDistances_test_noStopWords], f)
    print "Data stored", ((time.time()-time1)/60)

else:
    time1 = time.time()
    with open('calculated_variables/all_training_variables_yesNoStopWords.pickle', 'r') as f:  # Python 3: open(..., 'rb')
        train_1, train_2, labels, test_1, test_2, \
                     nb_words, tokenizer, textDistances, textDistances_test, \
                     train_1_noStopWords, train_2_noStopWords, test_1_noStopWords, test_2_noStopWords, \
                     nb_words_noStopWords, tokenizer_noStopWords, textDistances_noStopWords, textDistances_test_noStopWords = pickle.load(f)
    print "Data loaded", ((time.time()-time1)/60)
   
    
rand = np.random.permutation(labels.index)
train_1 = train_1[rand]
train_2 = train_2[rand]
labels = labels[rand]



