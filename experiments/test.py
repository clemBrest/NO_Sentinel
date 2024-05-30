#%%
import torch
import matplotlib.pyplot as plt
import numpy as np

from script.NO_Sentinel.utils.parser import Config
from script.NO_Sentinel.lightning.Lmodel import Lmodel
#%%

##################################################################
#       Arguments and model
##################################################################

device = torch.device('cpu')

args =  Config('config2.ini')

class RecurentNO(Lmodel):
    def __init__(self, model_name = 'NO', lr = 1e-4, loss_weights = {'reconstruction' : 0.1, 
                                        'linear' : 0.1, 
                                        'ortho_w' : 0,
                                        'ortho_conv': 0.1,
                                        'encode' : 0.1},
                                        future = 100,
                                        **pmodel):
        
        super(RecurentNO, self).__init__(model_name = model_name,
                            lr =  lr,
                            loss_weights = loss_weights,
                            future = future, 
                            **pmodel)



lightning_model = RecurentNO.load_from_checkpoint('/users/local/c23lacro/script/NO_Sentinel/runs/28051743/wavelet_res_linearSkip_lr1em4/epoch=993-val_loss=0.00.ckpt').to(device)

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

















# #%%
# import torch
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# import numpy as np

# from sentinel_data import SentinelDataset
# from parser import Config
# from Lmodel import RecurentN0
# #%%

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
# args = Config('config.ini')

# def criterion(x,y):
#     return torch.sqrt(torch.mean((x-y)**2))

# class model(RecurentN0):
#     def __init__(self):
#         super().__init__(n_modes=args.n_modes, 
#                     P_shape = args.P_shape, 
#                     Q_shape = args.Q_shape, 
#                     no_skip = args.no_skip,
#                     future = args.future,
#                     loss_weights=args.loss_weights,
#                     conv=args.conv,
#                     level=args.level,
#                     n_ino=args.n_ino)


# # model = RecurentN0(n_modes=args.n_modes, 
# #                     P_shape = args.P_shape, 
# #                     Q_shape = args.Q_shape, 
# #                     no_skip = args.no_skip,
# #                     future = args.future,
# #                     loss_weights=args.loss_weights,
# #                     conv = args.conv,
# #                     level = args.level,
# #                     n_ino = args.n_ino)

# lightning_model = model.load_from_checkpoint(
#                  '/users/local/c23lacro/script/NO_Sentinel/runs/24050743/0745/epoch=161-val_loss=0.00.ckpt'
#                 ).to(device)

# lightning_model.eval()
# #%%


# sentinel_data = SentinelDataset(path_data=args.path_data,
#                             n_train=2,
#                             train = False,
#                             size=args.size,
#                             future=1,
#                             missing_data=False)


# #%%

# i_image0 = 0
# mse_persistance = []
# mse_model = []
# mse_map = []
# pixel_tab_predict = []
# pixel_tab_sentinel = []

# image0 = sentinel_data[i_image0]['x'].to(device)
# input = image0.unsqueeze(0)


# for i in tqdm(range(i_image0, sentinel_data.__len__())):

#     res_output = lightning_model(input,1).squeeze(0)
#     # output = input 
#     # mse_persistance.append(criterion(image0, 
#     mse_persistance.append(criterion(image0, 
#                                      sentinel_data[i]['y'][0].to(device) )
#                                      .detach().cpu().numpy())
#     mse_model.append(criterion(res_output.to(device), 
#                                sentinel_data[i]['y'][0].to(device)).detach().cpu().numpy())
#     mse_map.append(np.sqrt(torch.mean( (res_output.squeeze(0).to(device) - sentinel_data[i]['x'].to(device))**2, dim = 0)
#                            .detach().cpu().numpy()))
#     pixel_tab_predict.append(res_output.to(device)[0,:,10,10].detach().cpu().numpy())
#     pixel_tab_sentinel.append(sentinel_data[i]['y'][0,:,10,10].detach().cpu().numpy())

#     input = res_output
# # %%


# #knowing each image append each 5 days
# time = np.arange(sentinel_data.__len__())*5
# plt.figure()
# #set vertical limit
# plt.title('RES net of 10 days trained recurent model')
# plt.plot(time[0:70], mse_persistance[0:70], label='persistance')
# plt.plot(time[0:70], mse_model[0:70], label='model')
# plt.legend()
# plt.xlabel('time (days)')
# plt.ylabel('RMSE')
# plt.show()
# #%%
# pixel_tab_predict = np.array(pixel_tab_predict)
# pixel_tab_sentinel = np.array(pixel_tab_sentinel)
# #plot the evolution of a pixel
# delta = 3000
# channel = 6
# plt.figure()
# # plt.ylim(0,1)

# plt.title('Evolution of pixel 10,10')
# plt.plot(time[i_image0:i_image0+delta], pixel_tab_predict[:delta,channel], label='predict')
# plt.plot(time[i_image0:i_image0+delta], pixel_tab_sentinel[:delta,channel], label='sentinel')
# plt.legend()
# plt.xlabel('time (days)')
# plt.ylabel('value')
# plt.show()


# # %%
# from matplotlib.animation import FuncAnimation
# from matplotlib import animation
# import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1 import make_axes_locatable

# fig, axs = plt.subplots(1, 2)  # Create 2 subplots side by side

# # axs[0].set_aspect('equal')
# # axs[1].set_aspect('equal')


# # Create the initial frames
# im1 = axs[0].imshow(sentinel_data[i_image0]['x'].detach().numpy()[[2,1,0],...].T*3 , animated=True)

# im2 = axs[1].imshow(mse_map[i_image0].T, animated=True, vmin=0, vmax=0.1)


# # Create new axes for the colorbars that are adjacent to the original axes
# divider1 = make_axes_locatable(axs[0])
# cax1 = divider1.append_axes("right", size="5%", pad=0.05)
# divider2 = make_axes_locatable(axs[1])
# cax2 = divider2.append_axes("right", size="5%", pad=0.05)


# # Add colorbars
# colorbar1 = plt.colorbar(im1, cax=cax1)
# colorbar1.remove()  # Remove the colorbar for the first image
# plt.colorbar(im2, cax=cax2)

# def update(i):
#     # Update each frame
#     im1.set_array(sentinel_data[i+i_image0]['x'].detach().numpy()[[2,1,0],...].T*3)

#     im2.set_array(mse_map[i].T)
#     #add a global title for the figure
#     fig.suptitle(f'RMSE map and Sentinel data at time {i*5} days', y = 0.8)
#     # # Update titles
#     # axs[0].set_title(f'MSE map at time {i*5} days')
#     # axs[1].set_title(f'Sentinel data at time {i*5} days')

#     return im1, im2,

# # Create the animations
# ani = FuncAnimation(fig, update, frames=np.arange(0, 100), blit=True)

# # plt.show()
# #save the video
# #%%
# ani.save('/users/local/c23lacro/script/2305/runs/22051638/1641/version_0/mse_map2.mp4', writer='ffmpeg', fps=2)

# # %%

# #take a pixels and plot the evolution of the pixel of channel 6
# #of sentinel data
# pixel = [10,10]
# for channel in range(10):
#     time = np.arange(sentinel_data.__len__())*5
#     plt.figure()
#     plt.title(f'Evolution of pixel {pixel} channel {channel} of Sentinel data')
#     plt.plot(time, [sentinel_data[i]['x'][channel,pixel[0],pixel[1]].detach().cpu().numpy() for i in range(sentinel_data.__len__())])
#     plt.xlabel('time (days)')
#     plt.ylabel('value')
#     plt.show()

# # %%

# %%
