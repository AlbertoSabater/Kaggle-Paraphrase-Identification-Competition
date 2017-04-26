from keras.callbacks import EarlyStopping, ModelCheckpoint

import numpy as np
import pandas as pd
import time
import models
import pickle
import os

np.random.seed(1337)  # for reproducibility


MAX_SEQUENCE_LENGTH = 30
earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.00001, verbose=1, mode='min', patience=3)


###########################################################

time1 = time.time()
with open('calculated_variables/all_training_variables_yesNoStopWords.pickle', 'r') as f:  # Python 3: open(..., 'rb')
    train_1, train_2, labels, test_1, test_2, \
                 nb_words, tokenizer, textDistances, textDistances_test, \
                 train_1_noStopWords, train_2_noStopWords, test_1_noStopWords, test_2_noStopWords, \
                 nb_words_noStopWords, tokenizer_noStopWords, textDistances_noStopWords, textDistances_test_noStopWords = pickle.load(f)
print "Data loaded", ((time.time()-time1)/60)
    
###########################################################

rand = np.random.permutation(labels.index)
train_1 = train_1[rand]
train_2 = train_2[rand]
labels = labels[rand]

train_1_noStopWords = train_1[rand]
train_2_noStopWords = train_2[rand]

###########################################################


EMBEDDING_LEN  = 300

'''
saving_file, model = models.model_v0_textDists(textDistances.shape[1], MAX_SEQUENCE_LENGTH, nb_words, EMBEDDING_LEN, num_lstm_lp=[512,512], 
                                         word_index=tokenizer.word_index, embeddings_file=4)
               
model.compile(loss='binary_crossentropy', optimizer='adam')
print "--", saving_file, "--"

checkpoint = ModelCheckpoint("Models/" + saving_file, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

callbacks_list = [earlyStopping]

time1 = time.time()
hist = model.fit([train_1, train_2, np.array(textDistances)], labels, epochs=200, batch_size=512, shuffle=True, validation_split=0.3, callbacks=callbacks_list)
print "Model trained", ((time.time()-time1)/60)
  

###########################################################
             
saving_file, model = models.model_v0_textDists(textDistances.shape[1], MAX_SEQUENCE_LENGTH, nb_words, EMBEDDING_LEN, num_lstm_lp=[512,512], 
                                         word_index=tokenizer.word_index, embeddings_file=4)
model.compile(loss='binary_crossentropy', optimizer='adam')
print "--", saving_file, "--"

checkpoint = ModelCheckpoint("Models/" + saving_file, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

callbacks_list = [earlyStopping]

time1 = time.time()
hist = model.fit([train_1, train_2, np.array(textDistances)], labels, epochs=200, batch_size=512, shuffle=True, validation_split=0.2, callbacks=callbacks_list)
print "Model trained", ((time.time()-time1)/60)


###########################################################

saving_file, model = models.model_v0_textDists(textDistances.shape[1], MAX_SEQUENCE_LENGTH, nb_words, EMBEDDING_LEN, num_lstm_lp=[512,512], 
                                         word_index=tokenizer.word_index, embeddings_file=4)
               
model.compile(loss='binary_crossentropy', optimizer='adam')
print "--", saving_file, "--"

checkpoint = ModelCheckpoint("Models/" + saving_file, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

callbacks_list = [earlyStopping]

time1 = time.time()
hist = model.fit([train_1, train_2, np.array(textDistances)], labels, epochs=200, batch_size=512, shuffle=True, validation_split=0.1, callbacks=callbacks_list)
print "Model trained", ((time.time()-time1)/60)


###########################################################
###########################################################



saving_file, model = models.model_v0_textDists(textDistances_noStopWords.shape[1], MAX_SEQUENCE_LENGTH, nb_words, EMBEDDING_LEN, num_lstm_lp=[512,512], 
                                         word_index=tokenizer.word_index, embeddings_file=4)
               
model.compile(loss='binary_crossentropy', optimizer='adam')
print "--", saving_file, "--"

checkpoint = ModelCheckpoint("Models/" + saving_file, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

callbacks_list = [earlyStopping]

time1 = time.time()
hist = model.fit([train_1, train_2, np.array(textDistances_noStopWords)], labels, epochs=200, batch_size=512, shuffle=True, validation_split=0.2, callbacks=callbacks_list)
print "Model trained", ((time.time()-time1)/60)

###########################################################

features = np.array(pd.concat([textDistances,textDistances_noStopWords], axis=1))

saving_file, model = models.model_v0_textDists(features.shape[1], MAX_SEQUENCE_LENGTH, nb_words, EMBEDDING_LEN, num_lstm_lp=[512,512], 
                                         word_index=tokenizer.word_index, embeddings_file=4)
               
model.compile(loss='binary_crossentropy', optimizer='adam')
print "--", saving_file, "--"

checkpoint = ModelCheckpoint("Models/" + saving_file, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

callbacks_list = [earlyStopping]

time1 = time.time()
hist = model.fit([train_1, train_2, features], labels, epochs=200, batch_size=512, shuffle=True, validation_split=0.2, callbacks=callbacks_list)
print "Model trained", ((time.time()-time1)/60)


###########################################################

features = np.array(pd.concat([textDistances,textDistances_noStopWords], axis=1))

saving_file, model = models.model_v0_textDists(features.shape[1], MAX_SEQUENCE_LENGTH, nb_words_noStopWords, EMBEDDING_LEN, num_lstm_lp=[512,512], 
                                         word_index=tokenizer_noStopWords.word_index, embeddings_file=4)
               
model.compile(loss='binary_crossentropy', optimizer='adam')
print "--", saving_file, "--"

checkpoint = ModelCheckpoint("Models/" + saving_file, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

callbacks_list = [earlyStopping]

time1 = time.time()
hist = model.fit([train_1_noStopWords, train_2_noStopWords, features], labels, epochs=200, batch_size=512, shuffle=True, validation_split=0.2, callbacks=callbacks_list)
print "Model trained", ((time.time()-time1)/60)

'''










def predictTest(save_file):
    files = os.listdir('Models')
    default_splited = saving_file.split('*')
    
    best_loss = 99999
    best_name = ''
    
    for name in files:
        current_splited = name.split('*')
        if default_splited[0] == current_splited[0] and default_splited[2] == current_splited[2]:
            if best_loss > float(current_splited[1]):
                best_loss = float(current_splited[1])
                best_name = name
                
    print "Best file saved:", best_name, "-" , best_loss
    
    filename, file_extension = os.path.splitext('Models/' + best_name)
    
    #################################################
    #################################################
    
    model.load_weights('Models/' + best_name)
    
    print "\nPredicting test values..."
    time1 = time.time()
    
    preds = model.predict([test_1, test_2])
    preds = np.round(preds).astype(int)
    
    submission = pd.DataFrame({'test_id': np.arange(len(preds)), 'is_duplicate': preds.ravel()})
    filename, file_extension = os.path.splitext(best_name)
    submission.to_csv('Predictions/test_' + filename + '.csv', columns=['test_id', 'is_duplicate'], index=False)
    
    print "Predictions obtained, processed and saved", ((time.time()-time1)/60)