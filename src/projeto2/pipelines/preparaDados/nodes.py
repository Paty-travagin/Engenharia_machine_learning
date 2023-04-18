"""
This is a boilerplate pipeline 'preparaaaab'
generated using Kedro 0.18.7
"""
import typer
import pandas as pd
import sklearn
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import GridSearchCV, RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import sklearn.metrics as metrics
import mlflow
from sklearn.metrics import mean_absolute_error, r2_score
import math
from xgboost import XGBRFRegressor 
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from pycaret.classification import ClassificationExperiment 
from pycaret.classification import *
from pycaret.regression import *
from sklearn.metrics import log_loss


mlflow.start_run()
##teste
def x_y(conformed):
    X = conformed.iloc[:, :-1] 
    return X
## teste
def y_test(conformed):
    y = conformed.iloc[:, 6:] 
    return y


#### filtar a coluna shot_type == 2PT Field Goal em parquert
def filter_short_type():
    data1= pd.read_csv('/infnet-machine/data.csv')
    data01= onformed =data1[['lat','shot_type','lon','minutes_remaining','period',
    'playoffs','shot_distance','shot_made_flag']]
    data01.dropna(inplace=True)
    filtrados = data1.loc[data1['shot_type'] == '2PT Field Goal']
    return filtrados

#### filtar a coluna shot_type == 3PT Field Goal parquert
def filter_short_type3pt():
    data1= pd.read_csv('/infnet-machine/data.csv')
    data01= onformed =data1[['lat','shot_type','lon','minutes_remaining','period',
    'playoffs','shot_distance','shot_made_flag']]
    data01.dropna(inplace=True)
    filtrados3 = data1.loc[data1['shot_type'] == '3PT Field Goal']
    return filtrados3

    #### filtar a coluna shot_type == 2PT Field Goal em mlflow pkl
def short_type2pt():
    data2= pd.read_csv('/infnet-machine/data.csv')
    data01= onformed =data2[['lat','shot_type','lon','minutes_remaining','period',
    'playoffs','shot_distance','shot_made_flag']]
    data01.dropna(inplace=True)
    filtrados2pt = data2.loc[data2['shot_type'] == '2PT Field Goal']
    return filtrados2pt

#### filtar a coluna shot_type == 3PT Field Goal mlflow pkl
def short_type3pt():
    data3= pd.read_csv('/infnet-machine/data.csv')
    data01= onformed =data3[['lat','shot_type','lon','minutes_remaining','period',
    'playoffs','shot_distance','shot_made_flag']]
    data01.dropna(inplace=True)
    filtrados3pt = data3.loc[data3['shot_type'] == '3PT Field Goal']
    return filtrados3pt

## teste csv
def comform_data():
    data2 = pd.read_csv('/infnet-machine/data.csv')
    dfconf =data2[['lat','lon','minutes_remaining','period','playoffs','shot_distance','shot_made_flag']].astype('float64')
    dfconf.dropna(inplace=True)
    #data['shot_made_flag'] = data['shot_made_flag'].astype('int').astype('float64')
    return dfconf


## fUNÇÃO PARA SELECIONAR AS VARIAVEIS DESEJADS 
def cof_data():
    data2 = pd.read_csv('/infnet-machine/data.csv')
    conformed =data2[['lat','lon','minutes_remaining','period','playoffs','shot_distance','shot_made_flag']].astype('float64')
    conformed.dropna(inplace=True)
    #data['shot_made_flag'] = data['shot_made_flag'].astype('int').astype('float64')
    return conformed



### FUNÇÃO PARA RETORNAR E SALVAR PARQUET DE TREINO E TEST
def svc(X,y):
    scaler = StandardScaler()
    scaler.fit(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    return X_train, X_test

    ### tamanho das metricas
def metrics(y_test,X_train):
    metricay= y_test.shape[0]
    metricax=  X_train.shape[0]
    return metricay,metricax

      ## Tamanho do dataset que utilizei na predições 
def dim_data(conformed):
    dimensao_data=  conformed.shape[0]
    return dimensao_data
    

    ### Algoritmos  classificação

def pycaret_mlsflow(conformed):
    s= setup(conformed, target ='shot_made_flag', session_id =123,
    n_jobs=-2, log_experiment='mlflow', experiment_name='classificador_nbakobe')
    exp= ClassificationExperiment()
    exp.setup(conformed, target ='shot_made_flag',
    session_id =123, n_jobs=-2, log_experiment='mlflow', experiment_name='classificador_nbakobe')
    exp.add_metric('logloss', 'Log Loss', log_loss, greater_is_better=False) 

    exp.compare_models()
    
    return exp


   ## Algoritmos  Regressão
def pycaret_classificador(conformed):
    s= setup(conformed, target ='shot_made_flag', session_id =123,
    n_jobs=-2, log_experiment='mlflow', experiment_name='regressor_nbakobe')

    exp= RegressionExperiment()
    exp.setup(conformed, target ='shot_made_flag',
    session_id =123, n_jobs=-2, log_experiment='mlflow', experiment_name='regressor_nbakobe')
    exp.add_metric('logloss', 'Log Loss', log_loss, greater_is_better=False) 
    exp.compare_models()
    return exp
'''
def logist_regre(conformed):
    exp= RegressionExperiment()
    exp.setup(conformed, target ='shot_made_flag',session_id =456, n_jobs=-2, log_experiment='mlflow', experiment_name='logist_regressor')
    exp.add_metric('logloss', 'Log Loss', log_loss, greater_is_better=False) 
    model_lr = exp.create_model('lr')

    return model_lr'''
    
def logist_regre(X_train):
    exp= RegressionExperiment()
    exp.setup(X_train,session_id =456, n_jobs=-2, log_experiment='mlflow', experiment_name='logist_regressor')
    exp.add_metric('logloss', 'Log Loss', log_loss, greater_is_better=False) 
    model_lr = exp.create_model('lr')

    return model_lr