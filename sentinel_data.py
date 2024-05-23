#%%
import numpy as np


import torch
from torch.utils.data import DataLoader
import sys



"""
This module contains the class SentinelDataset that is used to load the Sentinel data.
The data is loaded from a .npy file and rescaled to values between 0 and 1.

The class is a subclass of torch.utils.data.Dataset and implements the __len__ and __getitem__ methods.

__getitem__ returns a dictionary with the input and target data for the model.

Input data is the Sentinel data at time t and the differnce with t-1, and target data is the Sentinel data at time t+1 t+future and the finite difference.

The class has two modes: train and eval. In train mode, the data is loaded from the beginning of the dataset to n_train.
"""


class SentinelDataset(torch.utils.data.Dataset):
    """Custom Dataset class for PDE training data"""

    def __init__(self,device=torch.device('cpu'), 
                 train=True,
                 n_train=240,
                 size = 128,
                 future = 10,
                 path_data = '/users/local/c23lacro/data/Fontainebleau_interpolated_subdomain64.npy',
                 missing_data = False):
        
        # print('Class to load the Sentinel data')

        # #print of path of the current script
        # print('     Path of the current script')
        # print( sys.path[0])

        # print('     Loading sentinel data')
        self.sentinel_data0 = np.load(path_data)

        # We rescale the data to values between 0 and 1

        self.max = np.max(self.sentinel_data0)
        if np.max(self.sentinel_data0) > 1:
            self.sentinel_data0 /= self.max
            
        self.future = future
        self.size = size
        self.n_train = n_train
        self.missing_data = missing_data

        
        #TRASFORM TO TORCH
        self.train = train
        if self.train: 
            self.set_to_train()
        else:
            self.set_to_eval()

        self.shape = self.sentinel_data.shape
        self.nb_patch = self.shape[2]//self.size
        # print(f'     SentinelDataset shape: {self.__len__()}')

        self.device = device


        self.min = torch.min(self.sentinel_data)
        self.max = torch.max(self.sentinel_data)

    def set_to_eval(self):
        self.sentinel_data = torch.tensor(self.sentinel_data0).float()[self.n_train:,...]



    def set_to_train(self):
        self.sentinel_data = torch.tensor(self.sentinel_data0).float()[:self.n_train,...]



    def __len__(self):
        return (self.shape[0] -self.future)*self.nb_patch**2

    def __getitem__(self, index):
        """Get item method for PyTorch Dataset class"""


        #check if index is out of range
        if index >= self.__len__():
            raise IndexError('Index out of range')


        #get the subdomain index and the patch index and i,j position of the patch
        index_subdomain = index % (self.shape[0] -self.future)
        i_patch = index // (self.shape[0] -self.future)
        i,j = int(i_patch // self.nb_patch), int(i_patch % self.nb_patch)

        # print(f'index: {index}/{self.__len__()}, index_subdomain: {index_subdomain}/{self.shape[0] -self.future}, i_patch: {i_patch}/{self.nb_patch**2}, i: {i}, j: {j}')
        # sys.stdout.flush()

        if index_subdomain == 0:
            index_subdomain = 1
        

        #get index of the patch in the data
        x_index = slice(i*self.size, (i+1)*self.size)
        y_index = slice(j*self.size, (j+1)*self.size)
        tar_index = slice(index_subdomain + 1, index_subdomain + self.future + 1)
        tarm1_index = slice(index_subdomain, index_subdomain + self.future)
        
        # print(f'x_index: {x_index}, y_index: {y_index}, tar_index: {tar_index}')
        sys.stdout.flush()

        #add the difference beetween to consecutive time steps

        inpm1 = (self.sentinel_data[index_subdomain - 1, :,
                                    x_index, y_index])
        
        inp0 = (self.sentinel_data[index_subdomain, :,
                                  x_index, y_index])
        
        inp = torch.cat((inp0, inp0 - inpm1), dim=0)

        # if self.train:

        tarm1 = (self.sentinel_data[tarm1_index, :, 
                                    x_index, y_index])

        tar0 = (self.sentinel_data[tar_index, :,
                                    x_index, y_index])

        tar = torch.cat((tar0, tar0 - tarm1), dim=1)
        # else:
        #     tarm1 = (self.sentinel_data[index_subdomain, :, 
        #                                 x_index, y_index])

        #     tar0 = (self.sentinel_data[index_subdomain+1, :,
        #                                 x_index, y_index])

        #     tar = torch.cat((tar0, tar0 - tarm1), dim=0)

        if self.missing_data:
            mask = torch.ones_like(inp)
            
            x0 = np.random.randint(0,self.size) 
            y0 = np.random.randint(0,self.size)
            r = np.random.randint(0,self.size//4)

            for x in range(self.size):
                for y in range(self.size):
                    if (x-x0)**2 + (y-y0)**2 < r**2:
                        mask[:,x,y] = 0

            inp = inp * mask


        return {'x': inp.clone(), 'y': tar.clone()}


    
# %%

if __name__ == '__main__':
    Sdata = SentinelDataset(size=64,future=100, train = True, missing_data = False)

    print(Sdata[0]['x'].shape, Sdata[0]['y'].shape)



# %%
