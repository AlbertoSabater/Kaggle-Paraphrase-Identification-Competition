from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, multiply, Merge, merge
from keras.layers.merge import concatenate
from keras.models import Model, Sequential
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint

import os
import numpy as np
import time



# Call this method with load_embeddings=tokenizer.words_index to pretrain the embedding layer
# Mejor resultado con num_lstm_lp [512,512] y subiendo se podria mejorar
# fullText + [512, 512]                                             -> 
# text_noStopWords + [512, 512]                                     -> 
def model_v0(MAX_SEQUENCE_LENGTH, nb_words, EMBEDDING_LEN, num_lstm_lp, word_index=None, embeddings_file=None):
    
    saving_file = "model_v0_*{val_loss:.4f}*_" + str(MAX_SEQUENCE_LENGTH) + "_" + str(nb_words) + "_" + \
            str(EMBEDDING_LEN) + "_" + str(num_lstm_lp[0]) + "_" + str(num_lstm_lp[1])
    
    imp1 = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    imp2 = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

    if word_index!=None and embeddings_file!=None:
        if embeddings_file == 0:
            embedding_layer = Embedding(nb_words, EMBEDDING_LEN, input_length=MAX_SEQUENCE_LENGTH)
        elif embeddings_file == 1:     # embedding_len: 50, 100, 200, 300
            embedding_layer = prepareEmbeddingLayer('glove.6B.' + str(EMBEDDING_LEN) + 'd.txt', 
                                                    word_index, EMBEDDING_LEN, MAX_SEQUENCE_LENGTH)
            saving_file += "_le1"
        elif embeddings_file == 2:   # embedding_len: 25, 50, 100, 200, 
            embedding_layer = prepareEmbeddingLayer('glove.twitter.27B.' + str(EMBEDDING_LEN) + 'd.txt', 
                                                    word_index, EMBEDDING_LEN, MAX_SEQUENCE_LENGTH)
            saving_file += "_le2"
        elif embeddings_file ==3:
            embedding_layer = prepareEmbeddingLayer('glove.42B.300d.txt', word_index, 300, MAX_SEQUENCE_LENGTH)
            saving_file += "_le3"
        elif embeddings_file ==4:
            embedding_layer = prepareEmbeddingLayer('glove.840B.300d.txt', word_index, 300, MAX_SEQUENCE_LENGTH)
            saving_file += "_le4"
        else:
            raise ValueError('Fichero de embeddings no valido')
    else:
        embedding_layer = Embedding(nb_words, EMBEDDING_LEN, input_length=MAX_SEQUENCE_LENGTH)
        
    saving_file += ".hdf5"
    
    lstm_layer = LSTM(num_lstm_lp[0])
    
    model1 = embedding_layer(imp1)
    model2 = embedding_layer(imp2)
        
    model1 = lstm_layer(model1)
    model2 = lstm_layer(model2) 
    
#    model = merge([model1, model2])
    model = concatenate([model1, model2])
    model = BatchNormalization()(model)
    
    model = Dense(num_lstm_lp[1])(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    model = Dropout(0.2)(model)
    model = Dense(1)(model)
    model = BatchNormalization()(model)
    model = Activation('sigmoid')(model)
    
    model = Model([imp1, imp2], model)
   
    return saving_file, model



# fullText + [512, 512] + textDistances                                             -> 0.3497 - 0.3409 - 0.3334
# fullText + [512, 512] + textDistances_noStopWords                                 -> 0.3509
# fullText + [512, 512] + textDistances_noStopWords + textDistances                 -> 0.3503
# text_noStopWords + [512, 512] + textDistances_noStopWords + textDistances         -> 
def model_v0_textDists(dist_length, MAX_SEQUENCE_LENGTH, nb_words, EMBEDDING_LEN, num_lstm_lp, word_index=None, embeddings_file=None):
    
    saving_file = "model_v0_textDists*{val_loss:.4f}*_" + str(MAX_SEQUENCE_LENGTH) + "_" + str(nb_words) + "_" + \
            str(EMBEDDING_LEN) + "_" + str(num_lstm_lp[0]) + "_" + str(num_lstm_lp[1])
    
    imp1 = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    imp2 = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    imp3 = Input(shape=(dist_length,))

    if word_index!=None and embeddings_file!=None:
        if embeddings_file == 0:
            embedding_layer = Embedding(nb_words, EMBEDDING_LEN, input_length=MAX_SEQUENCE_LENGTH)
        elif embeddings_file == 1:     # embedding_len: 50, 100, 200, 300
            embedding_layer = prepareEmbeddingLayer('glove.6B.' + str(EMBEDDING_LEN) + 'd.txt', 
                                                    word_index, EMBEDDING_LEN, MAX_SEQUENCE_LENGTH)
            saving_file += "_le1"
        elif embeddings_file == 2:   # embedding_len: 25, 50, 100, 200, 
            embedding_layer = prepareEmbeddingLayer('glove.twitter.27B.' + str(EMBEDDING_LEN) + 'd.txt', 
                                                    word_index, EMBEDDING_LEN, MAX_SEQUENCE_LENGTH)
            saving_file += "_le2"
        elif embeddings_file ==3:
            embedding_layer = prepareEmbeddingLayer('glove.42B.300d.txt', word_index, 300, MAX_SEQUENCE_LENGTH)
            saving_file += "_le3"
        elif embeddings_file ==4:
            embedding_layer = prepareEmbeddingLayer('glove.840B.300d.txt', word_index, 300, MAX_SEQUENCE_LENGTH)
            saving_file += "_le4"
        else:
            raise ValueError('Fichero de embeddings no valido')
    else:
        embedding_layer = Embedding(nb_words, EMBEDDING_LEN, input_length=MAX_SEQUENCE_LENGTH)
        
    saving_file += ".hdf5"
    
    lstm_layer = LSTM(num_lstm_lp[0])
    
    model1 = embedding_layer(imp1)
    model2 = embedding_layer(imp2)
        
    model1 = lstm_layer(model1)
    model2 = lstm_layer(model2) 
    
    
#    features = Dense(28)(imp3)
    
#    model = concatenate([model1, model2, features], axis=1)
    model = concatenate([model1, model2, imp3], axis=1)
    model = BatchNormalization()(model)
    
    model = Dense(num_lstm_lp[1])(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    model = Dropout(0.2)(model)
    model = Dense(1)(model)
    model = BatchNormalization()(model)
    model = Activation('sigmoid')(model)
    
    model = Model([imp1, imp2, imp3], model)
    
    return saving_file, model




# https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
def prepareEmbeddingLayer(embeddings_file, word_index, embedding_len, max_sequence_len):
    print "Cargando embeddings:", embeddings_file
    time1 = time.time()
   
    embeddings_index = {}
    f = open(os.path.join('embeddings', embeddings_file))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_len))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
            
    embedding_layer = Embedding(len(word_index) + 1,
                            embedding_len,
                            weights=[embedding_matrix],
                            input_length=max_sequence_len,
                            trainable=False)
    
    print "Embeddings cargados", ((time.time()-time1)/60)
    
    return embedding_layer

