from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, multiply, Merge, merge
from keras.layers.merge import concatenate
from keras.models import Model, Sequential
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint

import os
import numpy as np
import time
import datetime



# Call this method with load_embeddings=tokenizer.words_index to pretrain the embedding layer
# Mejor resultado con num_lstm_lp [512,512] y subiendo se podria mejorar
# fullText + [512, 512]                                             -> 
# text_noStopWords + [512, 512]                                     -> 
# 0.3529
def model_v0(MAX_SEQUENCE_LENGTH, nb_words, EMBEDDING_LEN, num_lstm_lp, word_index=None, embeddings_file=4):
    
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



# fullText + [512, 512] + textDistances                                             -> 
# fullText + [512, 512] + textDistances_noStopWords                                 -> 
# fullText + [512, 512] + textDistances_noStopWords + textDistances                 -> 
# text_noStopWords + [512, 512] + textDistances_noStopWords + textDistances         -> 
# 0.3412
def model_v0_textDists(dist_length, MAX_SEQUENCE_LENGTH, nb_words, EMBEDDING_LEN, num_lstm_lp, word_index=None, embeddings_file=4):
    
    saving_file = "model_v0_textDists*{val_loss:.4f}*_" + str(MAX_SEQUENCE_LENGTH) + "_" + str(dist_length) + "_" + str(nb_words) + "_" + \
            str(EMBEDDING_LEN) + "_" + str(num_lstm_lp[0]) + "_" + str(num_lstm_lp[1])
    
    print saving_file
    
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


# fullText + [512, 512] + textDistances                                             -> 0.3497 - 0.3409 - 0.3334
# fullText + [512, 512] + textDistances_noStopWords                                 -> 0.3509
# fullText + [512, 512] + textDistances_noStopWords + textDistances                 -> 0.3462
# text_noStopWords + [512, 512] + textDistances_noStopWords + textDistances         -> 0.660
# 0.3401
def model_v1_textDists(dist_length, MAX_SEQUENCE_LENGTH, nb_words, EMBEDDING_LEN, num_lstm_lp, word_index=None, embeddings_file=4):
    
    saving_file = "model_v1_textDists*{val_loss:.4f}*_" + str(MAX_SEQUENCE_LENGTH) + "_" + str(dist_length) + "_" + str(nb_words) + "_" + \
            str(EMBEDDING_LEN) + "_" + str(num_lstm_lp[0]) + "_" + str(num_lstm_lp[1])
    
    print saving_file
    
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
    
    
    features = Dense(dist_length)(imp3)
    
    model = concatenate([model1, model2, features], axis=1)
#    model = concatenate([model1, model2, imp3], axis=1)
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



# 0.3375
def model_v1_1_textDists(dist_length, MAX_SEQUENCE_LENGTH, nb_words, EMBEDDING_LEN, num_lstm_lp, word_index=None, embeddings_file=4):
    
    saving_file = "model_v1_1_textDists*{val_loss:.4f}*_" + str(MAX_SEQUENCE_LENGTH) + "_" + str(dist_length) + "_" + str(nb_words) + "_" + \
            str(EMBEDDING_LEN) + "_" + str(num_lstm_lp[0]) + "_" + str(num_lstm_lp[1])
    
    print saving_file
    
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
    
    
#    features = Dense(dist_length)(imp3)
    
    model = concatenate([model1, model2], axis=1)
#    model = concatenate([model1, model2, imp3], axis=1)
    model = BatchNormalization()(model)
    
    model = Dense(num_lstm_lp[1])(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    model = Dropout(0.2)(model)

    model = concatenate([model, imp3], axis=1)
    
    model = Dense(num_lstm_lp[2])(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    model = Dropout(0.2)(model)
    
    model = Dense(1)(model)
    model = BatchNormalization()(model)
    model = Activation('sigmoid')(model)
    
    model = Model([imp1, imp2, imp3], model)
    
    return saving_file, model





# 0.3597
def model_v2_textDists(dist_length, MAX_SEQUENCE_LENGTH, nb_words, num_lstm_lp, word_index=None):
    
    saving_file = "model_v1_textDists*{val_loss:.4f}*_" + str(MAX_SEQUENCE_LENGTH) + "_" + str(dist_length) + "_" + str(nb_words) + "_" + \
            str(num_lstm_lp[0]) + "_" + str(num_lstm_lp[1])
    
    print saving_file
    
    imp1 = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    imp2 = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    imp3 = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    imp4 = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    imp5 = Input(shape=(dist_length,))


    embedding_layer = prepareEmbeddingLayer('glove.840B.300d.txt', word_index, 300, MAX_SEQUENCE_LENGTH)

    saving_file += ".hdf5"
    
    lstm_layer_raw = LSTM(num_lstm_lp[0])
    lstm_layer_clean = LSTM(num_lstm_lp[0])
    
    model1 = embedding_layer(imp1)
    model2 = embedding_layer(imp2)
    model3 = embedding_layer(imp3)
    model4 = embedding_layer(imp4)
        
    model1 = lstm_layer_raw(model1)
    model2 = lstm_layer_raw(model2) 
    model3 = lstm_layer_clean(model3)
    model4 = lstm_layer_clean(model4) 
    
    
    features = Dense(dist_length)(imp5)
    
    model = concatenate([model1, model2, model3, model4, features], axis=1)
#    model = concatenate([model1, model2, imp3], axis=1)
    model = BatchNormalization()(model)
    
    model = Dense(num_lstm_lp[1])(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    model = Dropout(0.2)(model)
    model = Dense(1)(model)
    model = BatchNormalization()(model)
    model = Activation('sigmoid')(model)
    
    model = Model([imp1, imp2, imp3, imp4, imp5], model)
    
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


# No dejar en blanco las palabras que estan fuera del corpus
def prepareEmbeddingLayer_v2(embeddings_file, word_index, embedding_len, max_sequence_len):
    print "Cargando embeddings:", embeddings_file, datetime.datetime.now().time().strftime("%H:%M:%S")
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
    
    print "Embeddings cargados", ((time.time()-time1)/60), datetime.datetime.now().time().strftime("%H:%M:%S")
    
    return embedding_layer






##Deriving the naive features
#for i in (1, 2):
#        transformed_sentences_train['question%s_tokens' % i] = train_sample['question%s' % i].apply(nltk.word_tokenize)
#        transformed_sentences_train['question%s_lowercase_tokens' % i] = transformed_sentences_train['question%s_tokens' % i].apply(convert_tokens_lower)
#        transformed_sentences_train['question%s_lowercase' % i] = transformed_sentences_train['question%s_lowercase_tokens' % i].apply(concatenate_tokens)
#        transformed_sentences_train['question%s_words' % i] = transformed_sentences_train['question%s_tokens' % i].apply(remove_stopwords)
#        transformed_sentences_train['question%s_pruned' % i] = transformed_sentences_train['question%s_words' % i].apply(concatenate_tokens)
#naive_similarity['similarity'] = np.vectorize(find_similarity)(train_sample['question1'], train_sample['question2'])
#naive_similarity['pruned_similarity'] = np.vectorize(find_similarity)(transformed_sentences_train['question1_pruned'], transformed_sentences_train['question2_pruned'])
#temp_features['common_tokens'] = np.vectorize(return_common_tokens)(transformed_sentences_train['question1_tokens'], transformed_sentences_train['question2_tokens'])