#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import time
#import models
import pickle
import os
import datetime
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn import linear_model

np.random.seed(1337)  # for reproducibility


# %% 

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


time1 = time.time()
with open('calculated_variables/all_training_variables_yesNoStopWords.pickle', 'r') as f:
    _, _, labels, _, _, \
     _, _, _, _, \
     _, _, _, _, \
     _, _, _, _ = pickle.load(f)
print "Data loaded", ((time.time()-time1)/60)


# %%

#X_train, X_test, y_train, y_test = train_test_split(train_data_preds, labels, test_size=0.3, random_state=0)
#
#scale = float(np.sum(y_train == 0)) / np.sum(y_train == 1)
#xgb = XGBClassifier(seed=132)
#
##w = float(len(y_train)) / (2 * np.bincount(y_train))
#
#xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric=metric, early_stopping_rounds=500)


# %% 
import sys
sys.path.insert(0, '/home/asabater/TFM/asabater/training_scripts')
import train_xgb

X_train, X_test, y_train, y_test = train_test_split(train_data_preds, labels, test_size=0.3, random_state=123)

xgb_params, total_tuning_time = train_xgb.tune_alg(X_train, y_train, X_test, y_test, balance=True, 
                                    init_max_delta_step=False, step_test=False, reduce_train_size=None)


metric = 'logloss'

alg = XGBClassifier(learning_rate = xgb_params['learning_rate'], 
                n_estimators = xgb_params['n_estimators'], 
                max_depth = xgb_params['max_depth'],
                min_child_weight = xgb_params['min_child_weight'], 
                gamma = xgb_params['gamma'], 
                subsample = xgb_params['subsample'],
                colsample_bytree = xgb_params['colsample_bytree'],
                objective = xgb_params['objective'], 
                nthread = xgb_params['nthread'], 
                scale_pos_weight = xgb_params['scale_pos_weight'], 
                max_delta_step = xgb_params['max_delta_step'],
                seed = 132)

alg.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric=metric, early_stopping_rounds=500)

pred_files = os.listdir('Predictions')
num_stack = len([ col for col in pred_files if col.startswith('stacked_') ])
sub_name = 'Predictions/stacked_xgb_' + str(num_stack) + '_' + metric + '_submission' + '.csv'


# %%

import sys
sys.path.insert(0, '/home/asabater/TFM/asabater/training_scripts')
import train_logReg

X_train, X_test, y_train, y_test = train_test_split(train_data_preds, labels, test_size=0.3, random_state=123)

logReg_params, total_tuning_time = train_logReg.tune_alg(X_train, y_train, X_test, y_test)


alg = linear_model.LogisticRegression(penalty=logReg_params['penalty'],
                                           C=logReg_params['C'],
                                           class_weight=logReg_params['class_weight'],
                                           solver=logReg_params['solver'],
                                           max_iter=logReg_params['max_iter'],
                                           random_state=logReg_params['random_state'])

alg.fit(X_train, y_train)


pred_files = os.listdir('Predictions')
num_stack = len([ col for col in pred_files if col.startswith('stacked_') ])
sub_name = 'Predictions/stacked_logReg_' + str(num_stack) + '_' + metric + '_submission' + '.csv'


# %%

predictions = alg.predict(test_data_preds)
submission = pd.DataFrame({'test_id': np.arange(len(predictions)), 'is_duplicate': predictions})
submission.to_csv(sub_name, columns=['test_id', 'is_duplicate'], index=False)
   


