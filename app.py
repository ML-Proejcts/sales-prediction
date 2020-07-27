# Importing essential libraries
from flask import Flask, render_template, request, Response
import pickle
import numpy as np
from datetime import date
import locale
import graphutil as graphutil
import util as util
from pandas.tseries.offsets import DateOffset
import pandas as pd
import os
from pathlib import Path
import io
import csv


model = pickle.load(open("xgb_reg.pkl", 'rb'))
print("** model downloaded **")
app = Flask(__name__)


@app.route('/')
def home():
    graphutil.graphForMonths()
    graphutil.graphForStores()
    return render_template('index.html', linePlotPath="\static\images\graph_month.png", samplePath="\static\images\graph_store.png")
    
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data_folder = Path("static/csv/")
        file = request.files.get('file')
        file_path = os.path.join(data_folder, file.filename)
        file.save(file_path)

        data = pd.read_csv(file_path)
        submission = pd.read_csv(file_path)
        data = util.convert_dates(data)
        data = pd.get_dummies(data, columns=["storeId"])
        predictions = model.predict(data)
        submission['sales']= predictions
        file_to_open = data_folder / "predicted-sales.csv"
        submission.to_csv(file_to_open,index=False)
        df = pd.read_csv(file_to_open)
        os.remove(file_path)
        return render_template('result.html', path_to_file = file_to_open, path_to_graph = graphutil.graphForPredictedMonths(df))

@app.route('/getCsv') # this is a job for GET, not POST
def plot_csv():
    fp = open(os.path.dirname(__file__)+'/predicted-monthly-sales.csv')
    csv = fp.read()
    return Response(
        csv,
        mimetype="text/csv",
        headers={"Content-disposition":
                 "attachment; filename=predicted-monthly-sales.csv"})
    
if __name__ == '__main__':
	app.run(debug=True)