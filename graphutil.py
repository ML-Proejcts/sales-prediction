import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import util as util

def read_data():
    df = pd.read_csv(os.path.dirname(__file__)+'/query_results.csv')
    return df

def graphForMonths():
    df= read_data()
    df = util.convert_dates(df)
    monthly_Months_sales = []
    monthly_Months_sales = df.groupby(['month'], as_index = True).agg({
    'grandSales': np.sum
    })
    monthly_Months_sales = monthly_Months_sales.reset_index()
    plt.subplots(figsize=(8, 6))
    sns.barplot(x="month", y="grandSales", hue="grandSales", data=monthly_Months_sales)
    my_path = os.path.dirname(__file__)+"/static/images"
    plt.title("Monthly sales")
    plt.xlabel('Month')
    plt.ylabel('Sales') 
    my_file = 'graph_month.png'
    plt.savefig(os.path.join(my_path, my_file))

def graphForStores():
    df= read_data()
    df = util.convert_dates(df)
    monthly_storeIds_sales = df.groupby(['storeId'], as_index = True).agg({
    'grandSales': np.sum
    })
    monthly_storeIds_sales = monthly_storeIds_sales.reset_index()
    plt.subplots(figsize=(8, 6))
    sns.barplot(x="storeId", y="grandSales", hue="grandSales", data=monthly_storeIds_sales)
    my_path = os.path.dirname(__file__)+"/static/images"
    plt.title("Store sales")
    plt.xlabel('Store')
    plt.ylabel('Sales') 
    my_file = 'graph_store.png'
    plt.savefig(os.path.join(my_path, my_file))

def graphForPredictedMonths(df):
    df = util.convert_dates(df)
    monthly_predicted_storeIds_sales = []
    monthly_predicted_storeIds_sales = df.groupby(['storeId'], as_index = True).agg({
    'sales': np.sum
    })
    monthly_predicted_storeIds_sales = monthly_predicted_storeIds_sales.reset_index()
    plt.subplots(figsize=(8, 6))
    sns.barplot(x="storeId", y="sales", hue="sales", data=monthly_predicted_storeIds_sales)
    my_path = os.path.dirname(__file__)+"/static/images"
    plt.title("Predicted sales")
    plt.xlabel('Store')
    plt.ylabel('Sales') 
    my_file = 'graph_predicted_store_'+ str(len(df))+'.png'
    plt.savefig(os.path.join(my_path, my_file))
    return "/static/images/"+my_file