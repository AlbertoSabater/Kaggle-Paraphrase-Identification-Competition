from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, multiply
from keras.layers.merge import concatenate
from keras.models import Model, Sequential
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint


def model_v0(MAX_SEQUENCE_LENGTH, nb_words, EMBEDDING_LEN, NUM_LSTM):
    imp1 = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    imp2 = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    
    model1 = Embedding(nb_words, EMBEDDING_LEN, input_length=MAX_SEQUENCE_LENGTH)(imp1)
    model2 = Embedding(nb_words, EMBEDDING_LEN, input_length=MAX_SEQUENCE_LENGTH)(imp2)
    
    model1 = LSTM(NUM_LSTM)(model1)
    model2 = LSTM(NUM_LSTM)(model2)    
    
    model = concatenate([model1, model2])
    model = Dropout(0.2)(model)
    
    model = Dense(128)(model)
    model = Activation('relu')(model)
    model = Dense(1)(model)
    model = Activation('sigmoid')(model)
    
    model = Model([imp1, imp2], model)
    
    return model