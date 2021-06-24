from flask import Flask, Response, jsonify, request
import pandas as pd
import os
from io import StringIO
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

training_data = pd.read_csv(os.path.join('Data', 'Iris data'))

trained_model = pd.read_pickle('classifier_decision_tree.pkl')

prediction_data = pd.read_csv(os.path.join('Data', 'Iris data Predict'))

@app.route('/')
def main():
    return {
        "Iris": "data",
    }

@app.route('/training_data')
def do():
    return Response(training_data.to_json(), mimetype='application/json')

@app.route('/prediction_data')
def cool():
    return Response(prediction_data.to_json(), mimetype='application/json')


@app.route('/predict')
def stuff():
    WTT = request.args.get('WTT')
    PTI = request.args.get('PTI')
    EQW = request.args.get('EQW')
    SBI = request.args.get('SBI')
    LQE = request.args.get('LQE')
    QWG = request.args.get('QWG')
    FDJ = request.args.get('FDJ')
    PJF = request.args.get('PJF')
    HQE = request.args.get('HQE')
    NXJ = request.args.get('NXJ')


    if(WTT and PTI and EQW and SBI and LQE and QWG and FDJ and PJF and HQE and NXJ):
               
        csv_string = ",".join([WTT,PTI,EQW,SBI,LQE,QWG,FDJ,PJF,HQE,NXJ])

        csv_data = StringIO(csv_string)

        test_attribute_names = ['WTT', 'PTI', 'EQW', 'SBI', 'LQE', 'QWG', 'FDJ', 'PJF', 'HQE', 'NXJ' ]
        prediction_data = pd.read_csv(csv_data, names=test_attribute_names)

        prediction = trained_model.predict(prediction_data)

        return {
            'result': prediction.item(0)
        }
    
    return Response('Please provide all neccessary parameters to get a prediction: zylinder, ps, gewicht, beschleunigung, baujahr', mimetype='application/json')