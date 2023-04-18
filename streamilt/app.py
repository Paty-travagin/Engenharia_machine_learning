import streamlit as st
import pickle
from flask import Flask, request
import requests
import pandas as pd
 #escrevendo um título na página
#st.title('Minha primeira aplicação :sunglasses:')
import joblib
from flask import Flask, jsonify, request
import numpy as np
import mlflow
from pycaret.classification import load_model

#st.write("ola mundo")
app = Flask(__name__)
model = load_model('..\projeto2/notebooks/my_best_pipeline.pkl','rb')

@app.route('/prever', methods=['POST'])

def prever():
    #data=[1.0,2.0,6.0,5.0,14.0,3.0]
    #data = {"lat":33.9203,"lon":-118.3438,"minutes_remaining":0.0,"period":2.0,"playoffs":0.0,"shot_distance":14.0}
    #result= model.predict(data)
    data = request.json
    predictions = model.predict(data)
    #return jsonify(data)
    return predictions.to_json(orient='records')
    #return jsonify(predictions.tolist())
    
if __name__ == '__main__':
    app.run(debug=True, port=5002)


#model= mlflow.sklearn.load_model("/infnet-machine/KEDRO/projeto2/data/02_intermidiate/regre_logist")

'''
app = Flask(__name__)
model = joblib.load('/infnet-machine/model lrnovo.pkl')
@app.route("/")
def home():
    return " meu trabalho deu certo"

@app.route('/prever', methods=['POST'])
def prever():
        #dados = {"lat":33.9203,"lon":-118.3438,"minutes_remaining":0.0,"period":2.0,"playoffs":0.0,"shot_distance":14.0}
        data= request.get_json()
        
        if data:
            if isinstance(data, dict):
                df_raw= pd.DataFrame(data, index[0])
            else:
                df_raw = pd.DataFrame(data,
                            columns= data[0].keys() )
            result= model.predict(df_raw)
            df_raw['prediction']= result
    # Retorne a resposta como JSON
            return df_raw.to_json(orient='records')
        return data

if __name__ == '__main__':
    app.run(debug=True, port=5005)


'''

