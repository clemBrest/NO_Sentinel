from .NOmodel import NO_model
from .MLPmodel import MLP_model

def Lmodel( **kwargs):
    data_dict = {'NO': NO_model, 
             'MLP': MLP_model
             }
    return data_dict.get(kwargs['model_name'])(**kwargs)

