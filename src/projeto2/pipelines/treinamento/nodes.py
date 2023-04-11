"""
This is a boilerplate pipeline 'treinamento'
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
#from.nodes import comform_data, x_y, svc
mlflow.autolog()
mlflow.sklearn.autolog()
from sklearn.model_selection import train_test_split
from pycaret.classification import ClassificationExperiment 
from pycaret.classification import *


######################################################################3
# MODELOS DE REGRESSÃO LinearRegression | XGBRFRegressor
########################################################################
def comform_data():
    data2 = pd.read_csv('/infnet-machine/data.csv')
    conformed =data2[['lat','lon','minutes_remaining','period','playoffs','shot_distance','shot_made_flag']].astype('float64')
    conformed.dropna(inplace=True)
    #data['shot_made_flag'] = data['shot_made_flag'].astype('int').astype('float64')
    return conformed

def lr_regression(X_train,y_train,X_test,y_test):
    lr = LinearRegression()
    lr.fit (X_train, y_train)
    test_pred_lin = lr.predict (X_test)
    lr.fit (X_train, y_train)
    rmse_lin = np.sqrt(metrics.mean_squared_error(y_test, test_pred_lin))
    r2_lin = metrics.r2_score (y_test, test_pred_lin)
    return rmse_lin, r2_lin





def XGBRFregresso(X_train,y_train,X_test,y_test):
    xgb= XGBRFRegressor(random_state=42)
    xgb.fit(X_train,y_train)
    xgb_predict = xgb.predict(X_test)
    mse = mean_absolute_error(y_test,xgb_predict)
    rmse= math.sqrt(mse)
    r2= r2_score(y_test,xgb_predict)
    return rmse, mse



def pycaret_mlsflow(data2):
    s= setup(data2, target ='shot_made_flag', session_id =123,
    n_jobs=-2, log_experiment='mlflow', experiment_name='regressor_nbakobe')
    exp= ClassificationExperiment()
    exp.setup(data2, target ='shot_made_flag',
    session_id =123, n_jobs=-2, log_experiment='mlflow', experiment_name='regressor_nbakobe')
    algort= exp.compare_models()
    return algort