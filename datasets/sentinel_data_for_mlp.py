#%%
import numpy as np
import torch


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

    def __init__(self,
                 train=True,
                 n_train=240,
                 size=64,
                 future=1,
                 path_data = '/users/local/c23lacro/data/Fontainebleau_interpolated_subdomain64.npy'):
      
        self.sentinel_data0 = np.load(path_data)

        # We rescale the data to values between 0 and 1

        self.max = np.max(self.sentinel_data0)
        if np.max(self.sentinel_data0) > 1:
            self.sentinel_data0 /= self.max
            
        self.n_train = n_train
        
        #TRASFORM TO TORCH
        self.train = train
        if self.train: 
            self.set_to_train()
        else:
            self.set_to_eval()

        self.shape = self.sentinel_data.shape

        self.min = torch.min(self.sentinel_data)
        self.max = torch.max(self.sentinel_data)

    def set_to_eval(self):
        self.sentinel_data = torch.tensor(self.sentinel_data0).float()[self.n_train:,...]

    def set_to_train(self):
        self.sentinel_data = torch.tensor(self.sentinel_data0).float()[:self.n_train,...]



    def __len__(self):
        return self.shape[0] *self.shape[-1]*self.shape[-2]

    def __getitem__(self, index):
        """Get item method for PyTorch Dataset class"""


        #check if index is out of range
        if index >= self.__len__():
            raise IndexError('Index out of range')


        #get the subdomain index and the patch index and i,j position of the patch
        t = index % self.shape[0]
        i_pixel = index // self.shape[0] 
        i,j = int(i_pixel // (self.shape[-1])), int(i_pixel % (self.shape[-1]))

        t = 1 if t == 0 else t


        tarm1 = (self.sentinel_data[t-1, :,i, j])

        tar0 = (self.sentinel_data[t , :, i, j])

        inp = torch.tensor([i, j, t], dtype=torch.float32)

        tar = torch.cat((tar0, tar0 - tarm1), dim=0)
        tar = tar.float()


        return {'X': inp.clone(), 'tar': tar.clone()}


    
# %%

if __name__ == '__main__':
    Sdata = SentinelDataset()

    print(Sdata[0]['X'].shape, Sdata[0]['tar'].shape)



# %%
# %%
