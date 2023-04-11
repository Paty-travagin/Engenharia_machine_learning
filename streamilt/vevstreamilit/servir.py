#import streamlit as st
import pickle
from flask import Flask, request
import requests
import pandas as pd
import joblib
# escrevendo um título na página
#st.title('Minha primeira aplicação :sunglasses:')

model=pickle.load(open('/infnet-machine/model (1).pkl','rb'))
'''
app= Flask(__name__ )

'''
@app.route( '/predict', methods=['GET'])

def texto():
    return 'site'

@app.route( '/predict', methods=['POST'])

def predict():
    user = request.form.get('nome')
    mail = request.form.get('email')

    return 'funcionou'
'''

@app.route( '/predict', methods=['POST'])

def predict():
    test_json =request.get_json()
    # coletar dados 
    
    if test_json:
        if isinstance( test_json, dict): # unique value
            df_row =pd.DataFrame(test_json, index=[0])
        else:
            df_row = pd.DataFrame( test_json, columns=test_json[0].keys())

# prediction
        pred = model.predict(df_row)
        df_row['prediction']= pred
        return df_row.to_json(orient='records')

    return 'erro'

if __name__== '__main__':
    app.run(host='127.0.0.1', port= 5002)


if __name__== '__main__':
    app.run(debug=True, port= 5002)

'''





from flask import Flask, jsonify, request


app = Flask(__name__)
model = joblib.load('/infnet-machine/model lr.pkl')


@app.route('/prever', methods=['POST'])
def prever():
    #dados = {'nome': 'João', 'idade': 30}
    dados = request.json  # Recebe os dados enviados como JSON
    # Faça o pré-processamento dos dados, se necessário
    # Faça a previsão usando o modelo carregado
    resultado = model.predict(dados)
    # Retorne a resposta como JSON
    return jsonify({'resultado': resultado.tolist()})
    #return jsonify(dados)

if __name__ == '__main__':
    app.run(debug=True, port= 5002)