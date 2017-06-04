from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import RMSprop

import numpy as np
import pandas as pd
import time
import models
import pickle
import os

np.random.seed(1337)  # for reproducibility


MAX_SEQUENCE_LENGTH = 30
earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.00001, verbose=1, mode='min', patience=2)


###########################################################

time1 = time.time()
with open('calculated_variables/all_training_variables_yesNoStopWords.pickle', 'r') as f:
    train_1, train_2, labels, test_1, test_2, \
     nb_words, tokenizer, textDistances, textDistances_test, \
     train_1_noStopWords, train_2_noStopWords, test_1_noStopWords, test_2_noStopWords, \
     nb_words_noStopWords, tokenizer_noStopWords, textDistances_noStopWords, textDistances_test_noStopWords = pickle.load(f)
print "Data loaded", ((time.time()-time1)/60)


###########################################################

BASE_DIR = 'ModelsStacked/'

files = os.listdir(BASE_DIR)

train_data = pd.DataFrame()
test_data = pd.DataFrame()

for model_dir in files:
    model_files = os.listdir(BASE_DIR + model_dir)
    if len(model_files) > 0:
        train_file = [ s for s in model_files if 'results_train' in s ][0]
        test_file = [ s for s in model_files if 'results_test' in s ][0]
        print train_file, test_file
        
        dt_train = pd.read_csv(BASE_DIR + model_dir + '/' + train_file)
        dt_test = pd.read_csv(BASE_DIR + model_dir + '/' + test_file)
        
        print train_data.shape, dt_train.shape
        
        dt_train.columns = [ col + "-" + model_dir for col in dt_train.columns ]
        dt_test.columns = [ col + "-" + model_dir for col in dt_test.columns ]
        
        train_data = pd.concat([train_data, dt_train], axis=1)
        test_data = pd.concat([test_data, dt_test], axis=1)
        
        
        
train_data_preds = train_data[[ col for col in train_data.columns if col.startswith('preds-') ]]
train_data_preds_round = train_data[[ col for col in train_data.columns if col.startswith('preds_round-') ]]
        
test_data_preds = test_data[[ col for col in test_data.columns if col.startswith('preds_test-') ]]
test_data_preds_round = test_data[[ col for col in test_data.columns if col.startswith('preds_round_test-') ]]

test_data_preds.columns = [ col.replace('_test','') for col in test_data_preds.columns ]
test_data_preds_round.columns = [ col.replace('_test','') for col in test_data_preds_round.columns ]
    

###########################################################

rand = np.random.permutation(labels.index)
train_1 = train_1[rand]
train_2 = train_2[rand]
labels = labels[rand]

train_data_preds = train_data_preds[rand]
train_data_preds_round = train_data_preds_round[rand]

textDistances = textDistances.iloc[rand]
textDistances_noStopWords = textDistances_noStopWords.iloc[rand]

train_1_noStopWords = train_1_noStopWords[rand]
train_2_noStopWords = train_2_noStopWords[rand]

###########################################################



def predictTest(save_file, data):
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
    
    preds = model.predict(data)
    preds = np.round(preds).astype(int)
    
    submission = pd.DataFrame({'test_id': np.arange(len(preds)), 'is_duplicate': preds.ravel()})
    filename, file_extension = os.path.splitext(best_name)
    submission.to_csv('Predictions/test_' + filename + '.csv', columns=['test_id', 'is_duplicate'], index=False)
    
    print "Predictions obtained, processed and saved", ((time.time()-time1)/60)
    
    
EMBEDDING_LEN  = 300


features = np.array(pd.concat([textDistances,textDistances_noStopWords], axis=1))

saving_file, model = models.model_v0_textDists(features.shape[1], MAX_SEQUENCE_LENGTH, nb_words, EMBEDDING_LEN, num_lstm_lp=[512,512,128], 
                                         word_index=tokenizer.word_index, embeddings_file=4)
               
model.compile(loss='binary_crossentropy', optimizer='adam')
print "--", saving_file, "--"

checkpoint = ModelCheckpoint("Models/" + saving_file, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

callbacks_list = [earlyStopping, checkpoint]

time1 = time.time()
hist = model.fit([train_1, train_2, features], labels, epochs=200, batch_size=512, shuffle=True, validation_split=0.2, callbacks=callbacks_list)
print "Model trained", ((time.time()-time1)/60)

features_test = np.array(pd.concat([textDistances_test,textDistances_test_noStopWords], axis=1))
predictTest(saving_file, [test_1, test_2, features_test])



features = np.array(pd.concat([textDistances,textDistances_noStopWords], axis=1))

saving_file, model = models.model_v1_textDists(train_data_preds.shape[1], MAX_SEQUENCE_LENGTH, nb_words, 300, num_lstm_lp=[512,512,128], 
                                         word_index=tokenizer.word_index)
               
#model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.0001))
model.compile(loss='binary_crossentropy', optimizer='adam')
print "--", saving_file, "--"

checkpoint = ModelCheckpoint("Models/" + saving_file, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

callbacks_list = [earlyStopping]

time1 = time.time()
hist = model.fit([train_1, train_2, train_data_preds], labels, epochs=200, batch_size=512, shuffle=True, validation_split=0.2, callbacks=callbacks_list)
print "Model trained", ((time.time()-time1)/60)

features_test = np.array(pd.concat([textDistances_test,textDistances_test_noStopWords], axis=1))
predictTest(saving_file, [test_1, test_2, test_data_preds])



features = np.array(pd.concat([textDistances,textDistances_noStopWords], axis=1))

saving_file, model = models.model_v0_textDists(features.shape[1], MAX_SEQUENCE_LENGTH, nb_words, 300, num_lstm_lp=[512,512,256], 
                                         word_index=tokenizer.word_index)
#               
#model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.0001))
model.compile(loss='binary_crossentropy', optimizer='adam')
print "--", saving_file, "--"

checkpoint = ModelCheckpoint("Models/" + saving_file, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

callbacks_list = [earlyStopping]

time1 = time.time()
hist = model.fit([train_1, train_2, features], labels, epochs=200, batch_size=512, shuffle=True, validation_split=0.2, callbacks=callbacks_list)
print "Model trained", ((time.time()-time1)/60)

features_test = np.array(pd.concat([textDistances_test,textDistances_test_noStopWords], axis=1))
predictTest(saving_file, [test_1, test_2, features_test])


saving_file, model = models.model_v0(MAX_SEQUENCE_LENGTH, nb_words, 300, num_lstm_lp=[512,512], 
                                         word_index=tokenizer.word_index, embeddings_file=0)
               
model.compile(loss='binary_crossentropy', optimizer='adam')

callbacks_list = [earlyStopping]

time1 = time.time()
hist = model.fit([train_1, train_2], labels, epochs=200, batch_size=512, shuffle=True, 
                 validation_split=0.2, callbacks=callbacks_list)

print "Model trained", ((time.time()-time1)/60)



