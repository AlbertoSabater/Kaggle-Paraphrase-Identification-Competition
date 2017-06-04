from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import RMSprop, Adam

import numpy as np
import pandas as pd
import time
import models
import pickle
import os
import datetime

np.random.seed(1337)  # for reproducibility

BASE_DIR = 'ModelsStacked/'


MAX_SEQUENCE_LENGTH = 30
EMBEDDING_LEN  = 300
earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.00001, verbose=1, mode='min', patience=5)


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
    if model_dir.startswith('model_'):
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

train_1_mod = train_1[rand]
train_2_mod = train_2[rand]
labels_mod = labels[rand]

train_data_preds = train_data_preds.values
train_data_preds_round = train_data_preds_round.values

train_data_preds_mod = train_data_preds[rand]
train_data_preds_round_mod = train_data_preds_round[rand]

test_data_preds = test_data_preds.values
test_data_preds_round = test_data_preds_round.values

textDistances_mod = textDistances.iloc[rand]
textDistances_noStopWords_mod = textDistances_noStopWords.iloc[rand]

train_1_noStopWords_mod = train_1_noStopWords[rand]
train_2_noStopWords_mod = train_2_noStopWords[rand]

features_mod = np.array(pd.concat([textDistances_mod,textDistances_noStopWords_mod], axis=1))   # Use in training
features_train = np.array(pd.concat([textDistances,textDistances_noStopWords], axis=1)) # Use in predict
features_test = np.array(pd.concat([textDistances_test,textDistances_test_noStopWords], axis=1))    # Use in predict


###########################################################
###########################################################

def createModelDir():
    files = os.listdir(BASE_DIR)
    dir_name = BASE_DIR + 'model_' + str(len(files)) + '/'
    os.makedirs(dir_name)
    print '\n\n\n#########################################################################'
    print '###############################  MODEL', len(files), ' ###############################'
    print '#########################################################################'
    return dir_name


def storePreds(model_dir, model, data_train, data_test, version="_1"):
    files = os.listdir(model_dir)

    best_loss = 99999
    best_name = ''
    
    # Get best model name
    for name in files:
        current_splited = name.split('*')
        if best_loss > float(current_splited[1]):
            best_loss = float(current_splited[1])
            best_name = name
                
    print "Best model name:", best_name, "-" , best_loss
    
    # Remove bad models
    for name in files:
        if name != best_name:
            os.remove(model_dir + name)

    # Load best model
    model.load_weights(model_dir + best_name)
    
    print "Predicting values...", datetime.datetime.now().time().strftime("%H:%M:%S")
    time1 = time.time()
 
    filename, file_extension = os.path.splitext(best_name)
    
#    preds = model.predict(data_train)
#    preds_round = np.round(preds).astype(int)
#    results_train = pd.DataFrame({'preds': preds.ravel(), 'preds_round': preds_round.ravel()})
#    results_train.to_csv(model_dir + filename + '_results_train' + version + '.csv', columns=['preds', 'preds_round'], index=False)
    
    print "Train data predicted and stored", datetime.datetime.now().time().strftime("%H:%M:%S")
 
    preds_test = model.predict(data_test)
    preds_round_test = np.round(preds_test).astype(int)
    results_test = pd.DataFrame({'preds_test': preds_test.ravel(), 'preds_round_test': preds_round_test.ravel()})    
    results_test.to_csv(model_dir + filename + '_results_test' + version + '.csv', columns=['preds_test', 'preds_round_test'], index=False)

    submission = pd.DataFrame({'test_id': np.arange(len(preds_test)), 'is_duplicate': preds_round_test.ravel()})
    submission.to_csv(model_dir + filename + '_submission' + version + '.csv', columns=['test_id', 'is_duplicate'], index=False)
    
    print "Predictions obtained, processed and saved", ((time.time()-time1)/60), datetime.datetime.now().time().strftime("%H:%M:%S")
    
    print "-- Taking a siesta"
    time.sleep(60*3)    # 5 mins
    print "-- Wake up!"
    
    
    
###########################################################
###########################################################
    

############################### MODEL X #######################################
model_dir = createModelDir()

saving_file, model = models.model_v3_textDists(features_mod.shape[1], MAX_SEQUENCE_LENGTH, nb_words, EMBEDDING_LEN, 
                                               num_lstm_lp=[512,512], word_index=tokenizer.word_index)
               
#model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.0001))
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0001))
#model.compile(loss='binary_crossentropy', optimizer='adam')

checkpoint = ModelCheckpoint(model_dir + saving_file, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [earlyStopping]

time1 = time.time()
hist = model.fit([train_1_mod, train_2_mod, features_mod], labels_mod, epochs=200, batch_size=512, shuffle=True, 
                 validation_split=0.2, callbacks=callbacks_list, verbose=2)

print "Model trained", ((time.time()-time1)/60)

storePreds(model_dir, model, [train_1, train_2, features_train], [test_1, test_2, features_test])
#storePreds(model_dir, model, [train_2, train_1, features_train], [test_2, test_1, features_test], version="_2")
###############################################################################


############################### STACKED #######################################
model_dir = createModelDir()

saving_file, model = models.model_v1_textDists(train_data_preds.shape[1], MAX_SEQUENCE_LENGTH, nb_words, EMBEDDING_LEN, 
                                               num_lstm_lp=[512,512,128], word_index=tokenizer.word_index)
               
#model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.0001))
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0001))
#model.compile(loss='binary_crossentropy', optimizer='adam')

checkpoint = ModelCheckpoint(model_dir + saving_file, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [earlyStopping, checkpoint]

time1 = time.time()
hist = model.fit([train_1_mod, train_2_mod, train_data_preds_mod], labels_mod, epochs=200, batch_size=512, shuffle=True, 
                 validation_split=0.2, callbacks=callbacks_list, verbose=2)

print "Model trained", ((time.time()-time1)/60)

storePreds(model_dir, model, [train_1, train_2, train_data_preds], [test_1, test_2, test_data_preds], version='_stacked')
#storePreds(model_dir, model, [train_2, train_1, features_train], [test_2, test_1, features_test], version="_2")
###############################################################################
    

############################### MODEL 0 #######################################
model_dir = createModelDir()

saving_file, model = models.model_v0(MAX_SEQUENCE_LENGTH, nb_words, EMBEDDING_LEN, num_lstm_lp=[512,512], 
                                         word_index=tokenizer.word_index, embeddings_file=4)
               
model.compile(loss='binary_crossentropy', optimizer='adam')

checkpoint = ModelCheckpoint(model_dir + saving_file, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [earlyStopping, checkpoint]

time1 = time.time()
hist = model.fit([train_1_mod, train_2_mod], labels_mod, epochs=200, batch_size=512, shuffle=True, 
                 validation_split=0.2, callbacks=callbacks_list, verbose=2)

print "Model trained", ((time.time()-time1)/60)

storePreds(model_dir, model, [train_1, train_2], [test_1, test_2])
###############################################################################


############################### MODEL 1 #######################################
model_dir = createModelDir()

saving_file, model = models.model_v0(MAX_SEQUENCE_LENGTH, nb_words, EMBEDDING_LEN, num_lstm_lp=[512,512], 
                                         word_index=tokenizer_noStopWords.word_index, embeddings_file=4)
               
model.compile(loss='binary_crossentropy', optimizer='adam')

checkpoint = ModelCheckpoint(model_dir + saving_file, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [earlyStopping, checkpoint]

time1 = time.time()
hist = model.fit([train_1_noStopWords_mod, train_2_noStopWords_mod], labels_mod, epochs=200, batch_size=512, shuffle=True, 
                 validation_split=0.2, callbacks=callbacks_list, verbose=2)

print "Model trained", ((time.time()-time1)/60)

storePreds(model_dir, model, [train_1_noStopWords, train_2_noStopWords], [test_1_noStopWords, test_2_noStopWords])
###############################################################################
###############################################################################



############################### MODEL 2 #######################################
model_dir = createModelDir()

saving_file, model = models.model_v0_textDists(features_mod.shape[1], MAX_SEQUENCE_LENGTH, nb_words, EMBEDDING_LEN, 
                                               num_lstm_lp=[512,512], word_index=tokenizer.word_index)
               
model.compile(loss='binary_crossentropy', optimizer='adam')

checkpoint = ModelCheckpoint(model_dir + saving_file, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [earlyStopping, checkpoint]

time1 = time.time()
hist = model.fit([train_1_mod, train_2_mod, features_mod], labels_mod, epochs=200, batch_size=512, shuffle=True, 
                 validation_split=0.2, callbacks=callbacks_list, verbose=2)

print "Model trained", ((time.time()-time1)/60)

storePreds(model_dir, model, [train_1, train_2, features_train], [test_1, test_2, features_test])
#storePreds(model_dir, model, [train_2, train_1, features_train], [test_2, test_1, features_test], version="_2")
###############################################################################


############################### MODEL 3 #######################################
model_dir = createModelDir()

saving_file, model = models.model_v0_textDists(features_mod.shape[1], MAX_SEQUENCE_LENGTH, nb_words, EMBEDDING_LEN, 
                                               num_lstm_lp=[512,512], word_index=tokenizer_noStopWords.word_index)
               
model.compile(loss='binary_crossentropy', optimizer='adam')

checkpoint = ModelCheckpoint(model_dir + saving_file, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [earlyStopping, checkpoint]

time1 = time.time()
hist = model.fit([train_1_noStopWords_mod, train_2_noStopWords_mod, features_mod], labels_mod, epochs=200, batch_size=512, shuffle=True, 
                 validation_split=0.2, callbacks=callbacks_list, verbose=2)

print "Model trained", ((time.time()-time1)/60)
storePreds(model_dir, model, [train_1_noStopWords, train_2_noStopWords, features_train], [test_1_noStopWords, test_2_noStopWords, features_test])
#storePreds(model_dir, model, [train_2_noStopWords, train_1_noStopWords, features_train], [test_2_noStopWords, test_1_noStopWords, features_test], version="_2")
###############################################################################
###############################################################################
###############################################################################


############################### MODEL 4 #######################################
model_dir = createModelDir()

saving_file, model = models.model_v1_textDists(features_mod.shape[1], MAX_SEQUENCE_LENGTH, nb_words, EMBEDDING_LEN, 
                                               num_lstm_lp=[512,512,128], word_index=tokenizer.word_index)
               
#model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.0001))
#model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0001))
model.compile(loss='binary_crossentropy', optimizer='adam')

checkpoint = ModelCheckpoint(model_dir + saving_file, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [earlyStopping, checkpoint]

time1 = time.time()
hist = model.fit([train_1_mod, train_2_mod, features_mod], labels_mod, epochs=200, batch_size=512, shuffle=True, 
                 validation_split=0.2, callbacks=callbacks_list, verbose=2)

print "Model trained", ((time.time()-time1)/60)

storePreds(model_dir, model, [train_1, train_2, features_train], [test_1, test_2, features_test])
#storePreds(model_dir, model, [train_2, train_1, features_train], [test_2, test_1, features_test], version="_2")
###############################################################################


############################### MODEL 5 #######################################
model_dir = createModelDir()

saving_file, model = models.model_v1_textDists(features_mod.shape[1], MAX_SEQUENCE_LENGTH, nb_words, EMBEDDING_LEN, 
                                               num_lstm_lp=[512,512], word_index=tokenizer_noStopWords.word_index)
               
model.compile(loss='binary_crossentropy', optimizer='adam')

checkpoint = ModelCheckpoint(model_dir + saving_file, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [earlyStopping, checkpoint]

time1 = time.time()
hist = model.fit([train_1_noStopWords_mod, train_2_noStopWords_mod, features_mod], labels_mod, epochs=200, batch_size=512, shuffle=True, 
                 validation_split=0.2, callbacks=callbacks_list, verbose=2)

print "Model trained", ((time.time()-time1)/60)

storePreds(model_dir, model, [train_1_noStopWords, train_2_noStopWords, features_train], [test_1_noStopWords, test_2_noStopWords, features_test])
#storePreds(model_dir, model, [train_2_noStopWords, train_1_noStopWords, features_train], [test_2_noStopWords, test_1_noStopWords, features_test], version="_2")
###############################################################################
###############################################################################
###############################################################################



############################### MODEL 6 #######################################
model_dir = createModelDir()

saving_file, model = models.model_v1_1_textDists(features_mod.shape[1], MAX_SEQUENCE_LENGTH, nb_words, EMBEDDING_LEN, 
                                               num_lstm_lp=[512,512,128], word_index=tokenizer.word_index)
               
#model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.0001))
#model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0001))
model.compile(loss='binary_crossentropy', optimizer='adam')

checkpoint = ModelCheckpoint(model_dir + saving_file, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [earlyStopping, checkpoint]

time1 = time.time()
hist = model.fit([train_1_mod, train_2_mod, features_mod], labels_mod, epochs=200, batch_size=512, shuffle=True, 
                 validation_split=0.2, callbacks=callbacks_list, verbose=2)

print "Model trained", ((time.time()-time1)/60)

storePreds(model_dir, model, [train_1, train_2, features_train], [test_1, test_2, features_test])
#storePreds(model_dir, model, [train_2, train_1, features_train], [test_2, test_1, features_test], version="_2")
###############################################################################


############################### MODEL 7 #######################################
model_dir = createModelDir()

saving_file, model = models.model_v1_1_textDists(features_mod.shape[1], MAX_SEQUENCE_LENGTH, nb_words, EMBEDDING_LEN, 
                                               num_lstm_lp=[512,512, 256], word_index=tokenizer_noStopWords.word_index)
               
model.compile(loss='binary_crossentropy', optimizer='adam')

checkpoint = ModelCheckpoint(model_dir + saving_file, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [earlyStopping, checkpoint]

time1 = time.time()
hist = model.fit([train_1_noStopWords_mod, train_2_noStopWords_mod, features_mod], labels_mod, epochs=200, batch_size=512, shuffle=True, 
                 validation_split=0.2, callbacks=callbacks_list, verbose=2)

print "Model trained", ((time.time()-time1)/60)

storePreds(model_dir, model, [train_1_noStopWords, train_2_noStopWords, features_train], [test_1_noStopWords, test_2_noStopWords, features_test])
#storePreds(model_dir, model, [train_2_noStopWords, train_1_noStopWords, features_train], [test_2_noStopWords, test_1_noStopWords, features_test], version="_2")
###############################################################################
###############################################################################
###############################################################################


############################### MODEL 8 #######################################
model_dir = createModelDir()

saving_file, model = models.model_v2_textDists(features_mod.shape[1], MAX_SEQUENCE_LENGTH, nb_words, 
                                               num_lstm_lp=[512,256], word_index=tokenizer.word_index)
               
model.compile(loss='binary_crossentropy', optimizer='adam')

checkpoint = ModelCheckpoint(model_dir + saving_file, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [earlyStopping, checkpoint]

time1 = time.time()
hist = model.fit([train_1_mod, train_2_mod, train_1_noStopWords_mod, train_2_noStopWords_mod, features_mod], labels_mod, epochs=200, batch_size=512, shuffle=True, 
                 validation_split=0.2, callbacks=callbacks_list, verbose=2)

print "Model trained", ((time.time()-time1)/60)

storePreds(model_dir, model, [train_1, train_2, train_1_noStopWords, train_2_noStopWords, features_train], [test_1, test_2, test_1_noStopWords, test_2_noStopWords, features_test])
#storePreds(model_dir, model, [train_2, train_1, train_2_noStopWords, train_1_noStopWords, features_train], [test_2, test_1, test_2_noStopWords, test_1_noStopWords, features_test], version="_2")
###############################################################################
###############################################################################
###############################################################################





############################### MODEL 9 #######################################
model_dir = createModelDir()

saving_file, model = models.model_v0(MAX_SEQUENCE_LENGTH, nb_words, EMBEDDING_LEN, num_lstm_lp=[512,512], 
                                         word_index=tokenizer.word_index, embeddings_file=0)
               
model.compile(loss='binary_crossentropy', optimizer='adam')

checkpoint = ModelCheckpoint(model_dir + saving_file, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [earlyStopping, checkpoint]

time1 = time.time()
hist = model.fit([train_1_mod, train_2_mod], labels_mod, epochs=200, batch_size=512, shuffle=True, 
                 validation_split=0.2, callbacks=callbacks_list, verbose=2)

print "Model trained", ((time.time()-time1)/60)

storePreds(model_dir, model, [train_1, train_2], [test_1, test_2])
#storePreds(model_dir, model, [train_2, train_1], [test_2, test_1], version="_2")
###############################################################################


############################### MODEL 10 #######################################
model_dir = createModelDir()

saving_file, model = models.model_v0(MAX_SEQUENCE_LENGTH, nb_words, EMBEDDING_LEN, num_lstm_lp=[512,512], 
                                         word_index=tokenizer_noStopWords.word_index, embeddings_file=0)
               
model.compile(loss='binary_crossentropy', optimizer='adam')

checkpoint = ModelCheckpoint(model_dir + saving_file, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [earlyStopping, checkpoint]

time1 = time.time()
hist = model.fit([train_1_noStopWords_mod, train_2_noStopWords_mod], labels_mod, epochs=200, batch_size=512, shuffle=True, 
                 validation_split=0.2, callbacks=callbacks_list, verbose=2)

print "Model trained", ((time.time()-time1)/60)

storePreds(model_dir, model, [train_1_noStopWords, train_2_noStopWords], [test_1_noStopWords, test_2_noStopWords])
#storePreds(model_dir, model, [train_2_noStopWords, train_1_noStopWords], [test_2_noStopWords, test_1_noStopWords], version="_2")
###############################################################################
###############################################################################
