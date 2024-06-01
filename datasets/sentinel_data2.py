
from datasets.sentinel_data import SentinelDataset as SentinelDatasetPDE
from datasets.sentinel_data_for_mlp import SentinelDataset as SentinelDatasetMLP
from datasets.sentinel_data_for_conv2d import SentinelDataset as SentinelDatasetLinConv




data_dict = {'NO': SentinelDatasetPDE, 
             'MLP': SentinelDatasetMLP,
            'LinConv': SentinelDatasetLinConv
             }

SentinelDataset = data_dict.get(args.model_name)