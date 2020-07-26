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
from sklearn.model_selection import train_test_split
import util as util
import pickle
from xgboost.sklearn import XGBRegressor

df = pd.read_csv(os.path.dirname(__file__)+'/query_results.csv')

df = util.convert_dates(df)

df = pd.get_dummies(df, columns=["storeId"])

x_train,x_test,y_train,y_test = train_test_split(df.drop('grandSales',axis=1),df.pop('grandSales'),random_state=123,test_size=0.2)


xbr = XGBRegressor(
             max_depth=50, min_child_weight=4, missing=None, n_estimators=100,
             objective='reg:linear')

xbr.fit(x_train, y_train)

# xbr = XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#              colsample_bynode=1, colsample_bytree=0.7, gamma=0,
#              importance_type='gain', learning_rate=0.01, max_delta_step=0,
#              max_depth=100, min_child_weight=4, missing=None, n_estimators=600,
#              n_jobs=-1, nthread=None, objective='reg:linear', random_state=6958,
#              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
#              silent=None, subsample=0.3, verbosity=0)

filename = 'model.pkl'

pickle.dump(xbr, open(filename, 'wb'))