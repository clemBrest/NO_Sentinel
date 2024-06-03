#%%
import torch
import matplotlib.pyplot as plt
import numpy as np

import sys 

sys.path.append('/users/local/c23lacro/script/NO_Sentinel/')
from utils.config import Config
from light.MLPmodel import MLP_model
#%%

##################################################################
#       Arguments and model
##################################################################

device = torch.device('cpu')

lightning_model = MLP_model.load_from_checkpoint('/users/local/c23lacro/script/NO_Sentinel/experiments/MLP/saving_path:experiments/MLP_n_epochs:1000_path_data:/users/local/c23lacro/data/Fontainebleau_interpolated_subdomain64.npy_batch_size:8192_lr:0.001_n_train:240_model_name:MLP_priorarch:[1, 256, 1024, 256, 10]_pixel:[20, 20]_activation:tanh/epoch=133-eval_loss_total=0.10.ckpt'
                                                 ).to(device)

lightning_model.eval()

path_data = '/users/local/c23lacro/data/Fontainebleau_interpolated_subdomain64.npy'

#%%

##################################################################
#       Data
##################################################################

sentinel_data = np.load(path_data)

if np.max(sentinel_data) > 1:
    sentinel_data = (sentinel_data - np.min(sentinel_data)) / (np.max(sentinel_data) - np.min(sentinel_data))
    # sentinel_data /= np.max(sentinel_data)

# sentinel_data = torch.tensor(sentinel_data).float()

# sentinel_data_diff = sentinel_data[1:,...] - sentinel_data[:-1,...]

# sentinel_data = torch.tensor(np.append( sentinel_data[1:,...],sentinel_data_diff, axis=1)).float()

# if args.model_name == 'Koopman_DeepOperatorNet' or args.model_name == 'KoopmanAE':
#     sentinel_data = sentinel_data.reshape(342,20,-1).permute(2,0,1)

#%%

##################################################################
#       Prediction
##################################################################

i_image0 = 0

prediction = []
sentinel = []
x = 20
y = 20

for t in range(sentinel_data.shape[0]):
    prediction.append(lightning_model(torch.tensor(t).float().unsqueeze(0)))
    sentinel.append(sentinel_data[t,:,x,y])



#%%
prediction_tensor = torch.stack(prediction).squeeze()
prediction_tensor.shape
#%%
plt.plot(prediction_tensor.detach().cpu().numpy()[:,6], label = 'prediction')
plt.plot(sentinel_data[:,6,x,y], label = 'sentinel')
plt.axvline(240, color = 'black', label = 'train-test split')
plt.legend()
plt.show()
# %%
