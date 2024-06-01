#%%
import torch
import matplotlib.pyplot as plt
import numpy as np

from script.NO_Sentinel.utils.config import Config
from light.Lmodel import Lmodel
#%%

##################################################################
#       Arguments and model
##################################################################

device = torch.device('cpu')

args =  Config('config2.ini')

class MLP(Lmodel):
    def __init__(self, model_name = 'MLP', lr = 1e-3 , loss_weights = None ,future = None ,
                                        **pmodel):
        
        super(MLP, self).__init__(model_name = model_name, 
                            **pmodel)



lightning_model = MLP.load_from_checkpoint('/users/local/c23lacro/script/NO_Sentinel/runs/30051743/MLP_[3, 256, 1024, 256, 20]_tanh_0.0001batch:8192/epoch=4-eval_loss_total=0.07.ckpt').to(device)

lightning_model.eval()

path_data = '/users/local/c23lacro/data/Fontainebleau_interpolated_subdomain64.npy'

#%%

##################################################################
#       Data
##################################################################

sentinel_data = np.load(path_data)

if np.max(sentinel_data) > 1:
    sentinel_data /= np.max(sentinel_data)


sentinel_data_diff = sentinel_data[1:,...] - sentinel_data[:-1,...]

sentinel_data = torch.tensor(np.append( sentinel_data[1:,...],sentinel_data_diff, axis=1)).float()

if args.model_name == 'Koopman_DeepOperatorNet' or args.model_name == 'KoopmanAE':
    sentinel_data = sentinel_data.reshape(342,20,-1).permute(2,0,1)

#%%

##################################################################
#       Prediction
##################################################################

i_image0 = 0

image0 = sentinel_data[i_image0].unsqueeze(0).to(device)

out,_ = lightning_model(image0,sentinel_data.shape[0]-i_image0)
out = out.squeeze(0)

#%%
sentinel_data.shape, image0.shape, out.shape

#%%

##################################################################
#       Metrics
##################################################################

mse_model = torch.sqrt(torch.mean((out-sentinel_data)**2, dim = (-3,-2,-1)))
mse_persistance = torch.sqrt(torch.mean((sentinel_data - image0)**2, dim = (-3,-2,-1)))

pixx ,pixy = 20, 20
channel = 5
pixel_predict = out[...,pixx,pixy].detach().cpu().numpy()
pixel_sentinel = sentinel_data[...,pixx,pixy].detach().cpu().numpy()
#%%
rmse_map = torch.sqrt(torch.mean((out-sentinel_data)**2, dim = 1))
rmse_map.shape

#%%
mse_model.shape, mse_persistance.shape, pixel_predict.shape, pixel_sentinel.shape

#%%
##################################################################
#       Plot
##################################################################
i = slice(0,170)
plt.plot(mse_model.detach().cpu().numpy()[i], label = 'model')
plt.plot(mse_persistance.detach().cpu().numpy()[i], label = 'persistance')
plt.title('RMSE between model and sentinel data')
plt.legend()
plt.show()

plt.plot(pixel_predict[:,channel], label = 'model')
plt.plot(pixel_sentinel[:,channel], label = 'sentinel')
plt.title(f'Prediction of one pixel {pixx},{pixy} channel {channel}')
plt.legend()
# %%

##################################################################
#       Animation
##################################################################

from matplotlib.animation import FuncAnimation
from matplotlib import animation
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

fig, axs = plt.subplots(1, 2)  # Create 2 subplots side by side
=
# Create the initial frames
im1 = axs[0].imshow(sentinel_data[i_image0].detach().numpy()[[2,1,0],...].T*3 , animated=True)

im2 = axs[1].imshow(rmse_map[i_image0].detach().numpy().T, animated=True, vmin=0, vmax=0.1)


# Create new axes for the colorbars that are adjacent to the original axes
divider1 = make_axes_locatable(axs[0])
cax1 = divider1.append_axes("right", size="5%", pad=0.05)
divider2 = make_axes_locatable(axs[1])
cax2 = divider2.append_axes("right", size="5%", pad=0.05)


# Add colorbars
colorbar1 = plt.colorbar(im1, cax=cax1)
colorbar1.remove()  # Remove the colorbar for the first image
plt.colorbar(im2, cax=cax2)

def update(i):
    # Update each frame
    im1.set_array(sentinel_data[i+i_image0].detach().numpy()[[2,1,0],...].T*3)

    im2.set_array(rmse_map[i].detach().numpy().T)
    #add a global title for the figure
    fig.suptitle(f'RMSE map and Sentinel data at time {i*5} days', y = 0.8)

    return im1, im2,

# Create the animations
ani = FuncAnimation(fig, update, frames=np.arange(0, 100), blit=True)

#save the video
#%%
# ani.save(f'{args.saving_path}/{args.str_name}/mse_map2.mp4', writer='ffmpeg', fps=2)
ani.save('/users/local/c23lacro/script/NO_Sentinel/runs/28051743/wavelet_res_linearSkip_lr1em4/mse_map.mp4',writer='ffmpeg', fps=2)
         

# %%
