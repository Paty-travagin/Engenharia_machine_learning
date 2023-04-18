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
from flask import request
import mlflow
import requests

# Exemplo básico de um aplicativo Streamlit
import streamlit as st

import streamlit as st
import pycaret
import pandas as pd
import pickle

from transformers import pipeline
from pycaret.classification import load_model







#importando as bibliotecas
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split































st.title('ENGENHARIA DE MACHINE LEARNING')
# Título do aplicativo
st.title("Nba Kobe")


modelo = load_model('\infnet-machine\KEDRO\projeto2\notebooks\my_best_pipeline.pkl')
# Interface do usuário para fazer predições
st.write("Faça uma predição:")
minutes_remaining = st.slider("minutos restantes", 4.0, 3.0, 5.0)
period = st.slider("periodo", 2.0, 4.5, 3.0)
playoffs = st.slider("jogos decisivos", 0.0, 2.0, 4.0)


# Realiza a predição
dados_predicao = pd.DataFrame({
    "minutes_remaining": [minutes_remaining],
    "period": [period],
    "playoffs": [playoffs],
    
})
predicao = modelo.predict(dados_predicao)

# Exibe o resultado da predição
st.write("Resultado da predição:")

st.write(f"A predição é: {predicao}")

