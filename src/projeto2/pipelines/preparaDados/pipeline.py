"""
This is a boilerplate pipeline 'preparaaaab'
generated using Kedro 0.18.7
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import comform_data,x_y,short_type2pt,short_type3pt
from .nodes import y_test, svc,metrics,filter_short_type3pt,cof_data
from .nodes import pycaret_mlsflow,pycaret_classificador, dim_data,filter_short_type

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
            node(
                func=comform_data,
                inputs=None,
                outputs='comforme_data',
                name='comform_data',
            ),
            node(
                func=x_y,
                inputs='comforme_data',
                outputs='x_test',
                name='x_y',

           ),
           node(
            func=y_test,
            inputs='comforme_data',
            outputs='y_teste',
            name='y_test',

           ),
           node(
            func=svc,
            inputs=['x_test','y_teste'],
            outputs=['Xtrain','ytrain'],
            name='svc'
           ),

           node(
            func=metrics,    
            name='metrics0',
            inputs=['Xtrain','ytrain'],
            outputs=['selectd_metricsy','selected_metricx']
         ),

         node(
            func=pycaret_mlsflow,
            name= 'algoritmos_classificacao',
            inputs='comforme_data',
            outputs='selectd_algoritmo_classi',
         ),
           node(
            func=pycaret_classificador,
            name= 'algoritmos_regressao',
            inputs='comforme_data',
            outputs='selectd_algo_regre',
         ),
          node(
            func=dim_data,
            name= 'dimensao_dataset',
            inputs='comforme_data',
            outputs='dime_dataset',
         ),
          node(
            func=filter_short_type,
            name= 'filter_shot_type',
            inputs=None,
            outputs='filter_shot_type',
         ),
           node(
            func=filter_short_type3pt,
            name= 'filter_short_type3pt',
            inputs=None,
            outputs='filt_shot_type3pt',
         ),
             node(
            func=short_type2pt,
            name= 'short_type2pt',
            inputs=None,
            outputs='shot_type2ptt',
         ),
           node(
            func=short_type3pt,
            name= 'short_type3pt',
            inputs=None,
            outputs='shot_type3ptt',
         ),
          node(
                func=cof_data,
                inputs=None,
                outputs='dados_limpos',
                name='comforme_data',
            ),
])
 


    
