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



#st.write("ola mundo")



model = joblib.load('/infnet-machine/model lr.pkl')

app = Flask(__name__)

@app.route('/prever', methods=['POST'])
def prever():
    #dados = {'nome': 'João', 'idade': 30}
    dados = request.json  # Recebe os dados enviados como JSON
    # Faça o pré-processamento dos dados, se necessário
    # Faça a previsão usando o modelo carregado
    dados_np = np.array(dados)
    resultado = model.predict(dados_np)
    # Retorne a resposta como JSON
    return jsonify({'resultado': resultado.tolist()})
    #return jsonify(dados)

if __name__ == '__main__':
    app.run(debug=True, port= 5002)