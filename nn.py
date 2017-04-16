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

import pickle
import cleaning_utils
import models
import time

np.random.seed(1337)  # for reproducibility


# TO-DO: define max words
MAX_NB_WORDS = 600000           # 424706
MAX_SEQUENCE_LENGTH = 30
EMBEDDING_LEN  = 265

NUM_LSTM = 128



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

#q1 = cleaning_utils.removeStopWords(q1)
#q2 = cleaning_utils.removeStopWords(q2)
#q1_test = cleaning_utils.removeStopWords(q1_test)
#q2_test = cleaning_utils.removeStopWords(q2_test)

print "Data cleaned", ((time.time()-time1)/60)


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


print "Data preprocessed", ((time.time()-time1)/60)


rand = np.random.permutation(q1.index)
train_1 = train_1[rand]
train_2 = train_2[rand]
labels = labels[rand]


#################################################
#################################################


model = models.model_v0(MAX_SEQUENCE_LENGTH, nb_words, EMBEDDING_LEN, NUM_LSTM)
model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam')

hist = model.fit([train_1, train_2], labels, epochs=200, batch_size=512, shuffle=True, validation_split=0.2)

preds = model.predict([test_1, test_2])


preds = np.round(preds).astype(int)

submission = pd.DataFrame({'test_id': np.arange(len(preds)), 'is_duplicate': preds.ravel()})
#submission = submission[['test_id', 'is_duplicate']]
submission.to_csv('submission.csv', columns=['test_id', 'is_duplicate'], index=False)



