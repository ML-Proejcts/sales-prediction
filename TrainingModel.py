# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Any results you write to the current directory are saved as output.

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import util as util
import pickle
from sklearn.preprocessing import OneHotEncoder
from pycaret.regression import *


df = pd.read_csv(os.path.dirname(__file__)+'/query_results.csv')

df = util.convert_dates(df)

enc = OneHotEncoder()
# enc_df = pd.DataFrame(enc.fit_transform(df[['storeId']]).toarray())

df = pd.get_dummies(df, columns=["storeId"])

# df = df.join(enc_df)
print("dffffffffffffffffffffffffff")
print(df)

x_train,x_test,y_train,y_test = train_test_split(df.drop('grandSales',axis=1),df.pop('grandSales'),random_state=123,test_size=0.2)

def findMaxDepth(dtrain, num_boost_round):
    gridsearch_params = [
    (min_samples_split, min_samples_leaf)
    for min_samples_split in [10,25,50,75,100,125]
    for min_samples_leaf in  [10,25,50,75,100,125]
    ]

    min_mae = float("Inf")
    best_params = None
    for min_samples_split, min_samples_leaf in gridsearch_params:
        print("CV with max_depth={}, min_child_weight={}".format(
                                min_samples_split,
                                min_samples_leaf))
        # Update our parameters
        params = {}
        params['eta'] = 0.2
        params['max_depth'] = 5
        params['min_child_weight'] = 5
        # Run CV
        cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            seed=42,
            nfold=5,
            metrics={'rmse'},
            early_stopping_rounds=10
        )
        # Update best MAE
        mean_mae = cv_results['test-rmse-mean'].min()
        boost_rounds = cv_results['test-rmse-mean'].argmin()
        print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))
        if mean_mae < min_mae:
            min_mae = mean_mae
            best_params = (min_samples_split,min_samples_leaf)
    print("Best params: {}, {}, MAE: {}".format(best_params[0], best_params[1], min_mae)) 

def findEta(dtrain, num_boost_round):
    min_rmse = float("Inf")
    best_params = None
    for eta in [.3, .2, .1, .05, .01,.001]:
        print("CV with eta={}".format(eta))
        # We update our parameters
        params = {}
        params['eta'] = eta
        params['max_depth'] = 5
        params['min_child_weight'] = 5
        # Run and time CV
        cv_results = xgb.cv(
                params,
                dtrain,
                num_boost_round=num_boost_round,
                seed=42,
                nfold=5,
                metrics=['rmse'],
                early_stopping_rounds=10
            )
        # Update best score
        mean_rmse = cv_results['test-rmse-mean'].min()
        boost_rounds = cv_results['test-rmse-mean'].argmin()
        print("\trmse {} for {} rounds\n".format(mean_rmse, boost_rounds))
        if mean_rmse < min_rmse:
            min_rmse = mean_rmse
            best_params = eta
    print("Best params: {}, rmse: {}".format(best_params, min_rmse))


def XGBmodel(x_train,x_test,y_train,y_test):
    matrix_train = xgb.DMatrix(x_train,label=y_train)
    matrix_test = xgb.DMatrix(x_test,label=y_test)
    progress = dict()
    eval_set = [(x_train, y_train), (x_test, y_test)]
    params = {'max_depth': 100,
              'eta':0.3,
              'subsample':0.3,
              'min_child_weight': 4,
              'min_samples_split':25,
              'min_samples_leaf':25,
              'objective':'reg:linear',
              'eval_metric': 'rmse'
    }
    
    # findMaxDepth(matrix_train,1500)
    # cv_results = xgb.cv(
    #     params,
    #     matrix_train,
    #     num_boost_round=1500,
    #     seed=42,
    #     nfold=5,
    #     metrics={'rmse'},
    #     early_stopping_rounds=20,
    #     verbose_eval=True
    # )
    
    # print("********* cross val score MIN *************")
    # print(cv_results['train-rmse-mean'].min())
    # print(cv_results['test-rmse-mean'].min())
    
    # print("********* cross val score Mean *************")
    # print(cv_results['train-rmse-mean'].mean())
    # print(cv_results['test-rmse-mean'].mean())
    
    # range = np.arange(start=0, stop=len(cv_results), step=1)
    # plt.plot(range,cv_results['train-rmse-mean'].values,color='r')
    # plt.plot(range,cv_results['test-rmse-mean'].values, color='b')
    # plt.show()
    
    model=xgb.train(params 
                    ,dtrain=matrix_train,num_boost_round=2500, 
                    early_stopping_rounds=20,evals_result=progress,evals=[(matrix_train,'test')])
    
    print("********* eval MIN *************")
    print(min(progress['test']['rmse']))
  
    return model

# model=XGBmodel(x_train,x_test,y_train,y_test)

from xgboost.sklearn import XGBRegressor

xbr = XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=0.7, gamma=0,
             importance_type='gain', learning_rate=0.01, max_delta_step=0,
             max_depth=100, min_child_weight=4, missing=None, n_estimators=600,
             n_jobs=-1, nthread=None, objective='reg:linear', random_state=6958,
             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
             silent=None, subsample=0.3, verbosity=0)

xbr.fit(x_train, y_train)

predict=xbr.predict(x_test)

# from xgboost import plot_importance
# plot_importance(xbr)
# plt.show()

print('************************** XBR Evualtion ********************')
x_test['predict'] = predict
rmse = np.sqrt(mean_squared_error(y_test, predict))
print("RMSE" , rmse)
print("MAE", mean_absolute_error(y_test, predict))
print("r2_score" , r2_score(y_test, predict))
print(xbr.score(x_train, y_train))

# xbr.plot_importance(model)
# plt.rcParams['figure.figsize'] = [5, 5]
# plt.show()

  


# print("********************* XGB MODEL *******************")
# print("best_ntree_limit",model.best_ntree_limit)
# print("best_score",model.best_score)
# print("best_iteration",model.best_iteration)

# dtest = xgb.DMatrix(x_test, label=y_test, feature_names=list(x_test.columns))
# y_pred = model.predict(dtest)
# x_test['pred'] = y_pred
# # x_test['predict'] = predict
# print(x_test.to_csv(os.path.dirname(__file__)+'/xtest_results.csv'))

# print('************************** XGB Evualtion ********************')
# rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# print("RMSE" , rmse)
# print("MAE", mean_absolute_error(y_test, y_pred))
# print("r2_score" , r2_score(y_test, y_pred))

def merge(x,y,col,col_name):
    print('******************************* merge **************************')
    print(x)
    print(y)
    x =pd.merge(x, y, how='left', on=None, left_on=col, right_on=col,
            left_index=False, right_index=False, sort=True,
             copy=True, indicator=False,validate=None)
    x=x.rename(columns={'grandSales':col_name})
    return x

x_pred= pd.read_csv(os.path.dirname(__file__)+'/query_test.csv')
submission= pd.read_csv(os.path.dirname(__file__)+'/query_test.csv')

x_pred = util.convert_dates(x_pred)

x_pred = pd.get_dummies(x_pred, columns=["storeId"])

if 'storeId_169' not in x_pred.columns:
    x_pred['storeId_169'] = 0
    
if 'storeId_11' not in x_pred.columns:
    x_pred['storeId_11'] = 0

# x_pred=merge(x_pred, day_of_year_avg,['ItemId','storeId','day_of_year'],'day_of_year_avg')
print(x_pred)
predictions = xbr.predict(x_pred)
submission['sales']= predictions.round(2)
print("***************** new_prediction ****************")
xgb1772020 = load_model('xgb1772020')
x_pred['storeId'] = submission['storeId']
x_pred['businessDate'] = submission['businessDate']

new_prediction = predict_model(xgb1772020, data=x_pred)
print(new_prediction)
submission['pycaretsales'] = new_prediction['Label']

submission.to_csv('submission.csv',index=False)