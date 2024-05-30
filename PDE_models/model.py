#%%
import torch
from torch import nn
import torch.nn.functional as F
from .convolution import SpectralConv, WaveConv
#%%

class NO(nn.Module):
    def __init__(self, **pmodel):
        """
        args:
        pmodel (dict): dictionary containing the parameters of the model as

        'P_shape' : list of integers, shape of the encoder
        'Q_shape' : list of integers, shape of the decoder
        'n_modes' : int, number of modes for the spectral convolutions
        'no_skip' : str, type of skip connection
        'conv' : str, type of convolution
        'level' : int, level of wavelet decomposition
        'n_ino' : int, number of INO blocks
        'residual' : bool, residual connection
        'nonlinearity' : function, nonlinearity
        """
        
        super(NO, self).__init__()

        self.pmodel = pmodel

        self.encoder = MLP(pmodel['P_shape'])

        self.one_step = NO_Block(**pmodel)
        
        self.decoder = MLP(pmodel['Q_shape'])
    
    def forward(self, x, n=1):
            
        phis = []
        x_advanceds = []

        phis.append(self.encoder(x))

        for _ in range(n):

            phis.append(self.one_step(phis[-1]))

            if self.pmodel['residual']:
                x_advanceds.append(self.decoder(phis[-1] + phis[0]))
            else:
                x_advanceds.append(self.decoder(phis[-1]))

        return torch.stack(x_advanceds, dim=1),  torch.stack(phis, dim=1)


 
class NO_Block(nn.Module):
    def __init__(self, **pmodel):
        super(NO_Block, self).__init__()

        self.pmodel = pmodel
        self.channels = pmodel['P_shape'][-1]   
        self.conv = pmodel['conv']

        if pmodel['activation'] == 'gelu':
            self.activation = F.gelu
        elif pmodel['activation'] == 'relu':
            self.activation = F.relu
        elif pmodel['activation'] == 'Linear':
            self.activation = Identity()
        else:
            raise ValueError(f"activation {pmodel['activation']} not recognized")


        if self.conv == 'fourier':
            self.convs = SpectralConv( self.channels, pmodel['n_modes'])
        elif self.conv == 'wavelet':
            self.convs = WaveConv(self.channels, pmodel['n_modes'], level = pmodel['level'])

        elif self.conv == 'FilterConvolution':
            self.convs = nn.Conv2d(self.channels, self.channels, kernel_size = pmodel['kernel'], stride = pmodel['stride'], padding = pmodel['padding'])
            
        else:
            raise ValueError(f"Convolution {self.conv} not recognized")

        if pmodel['no_skip'] == 'linear':
            self.skip = nn.Conv2d(self.channels, self.channels, kernel_size = (1,1), stride = (1,1))
        elif pmodel['no_skip'] == 'identity':
            self.skip = Identity()   
        else:
            self.skip = 'None'
            print('No skip connection')

    def forward(self, x):

        x_old = x.clone()

        x_convs = self.convs(x_old)

        if self.skip == 'None':
            x = x_convs
        else:
            x_skip = self.skip(x_old)
            x = x_convs + x_skip

        x = self.activation(x)
        return x
    

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
    

class MLP(nn.Module):
    def __init__(self, shape = [20,256,32], nonlinearity = F.gelu):
        super(MLP, self).__init__()
        self.shape = shape
        self.nonlinearity = nonlinearity

        self.fcs = nn.ModuleList()
        for i in range(len(self.shape)-1):
            self.fcs.append(nn.Conv2d(self.shape[i], self.shape[i+1], kernel_size = (1,1), stride = (1,1)))

    def forward(self, x):
        for layer in self.fcs[:-1]:
            x = self.nonlinearity(layer(x))
        x = self.fcs[-1](x)
        return x
  
