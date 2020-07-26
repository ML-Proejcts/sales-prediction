import pandas as pd

def convert_dates(x):
    x['businessDate']=pd.to_datetime(x['businessDate'])
    x['month']=x['businessDate'].dt.month
    x['is_month_start']=x['businessDate'].dt.is_month_start
    x['is_month_end']=x['businessDate'].dt.is_month_end
    x['year']=x['businessDate'].dt.year
    x['dayofweek']=x['businessDate'].dt.dayofweek
    x['quarter'] = x['businessDate'].apply(lambda x: x.quarter)
    x['week_of_year'] = x['businessDate'].apply(lambda x: x.weekofyear)
    x['day_of_year'] = x['businessDate'].apply(lambda x: x.dayofyear)
    x['Is_Mon'] = (x.dayofweek == 0) *1
    x['Is_Tue'] = (x.dayofweek == 1) *1
    x['Is_Wed'] = (x.dayofweek == 2) *1
    x['Is_Thu'] = (x.dayofweek == 3) *1
    x['Is_Fri'] = (x.dayofweek == 4) *1
    x['Is_Sat'] = (x.dayofweek == 5) *1
    x['Is_Sun'] = (x.dayofweek == 6) *1
    x['Is_wknd'] = x.dayofweek // 4
    x.pop('businessDate')
    # x.pop('year')
    return x

def merge(x,y,col,col_name):
    x =pd.merge(x, y, how='left', on=None, left_on=col, right_on=col,
            left_index=False, right_index=False, sort=True,
             copy=True, indicator=False,validate=None)
    x=x.rename(columns={'sales':col_name})
    return x