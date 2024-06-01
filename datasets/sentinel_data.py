
from datasets.sentinel_data_for_NO import SentinelDataset as SentinelDatasetPDE
from datasets.sentinel_data_for_mlp import SentinelDataset as SentinelDatasetMLP
from datasets.sentinel_data_for_conv2d import SentinelDataset as SentinelDatasetLinConv


def SentinelDataset(train, **kwargs):
    data_dict = {'NO': SentinelDatasetPDE, 
             'MLP': SentinelDatasetMLP,
            'LinConv': SentinelDatasetLinConv
             }
    return data_dict.get(kwargs['model_name'])(**kwargs, train = train)

