#%%
import torch
from torch import nn
import torch.nn.functional as F
from pytorch_wavelets import DWT, IDWT 
import numpy as np
#%%

class NO(nn.Module):
    def __init__(self, n_modes, 
                 P_shape = [20,256,32], 
                 Q_shape = [32,256,20],
                   no_skip = 'linear', 
                   conv = 'fourier',
                   level = None):
        
        super(NO, self).__init__()

        self.n_modes = n_modes
        self.Q_shape = Q_shape
        self.P_shape = P_shape
        self.no_skip = no_skip
        self.conv = conv
        self.level = level

        self.hidden_channels = self.P_shape[-1]

        self.encoder = MLP(self.P_shape)
        self.NO_Block = NO_Block(n_modes = self.n_modes, 
                                  channels=self.hidden_channels, 
                                  skip=self.no_skip, 
                                  conv = self.conv,
                                    level = self.level)
        
        self.decoder = MLP(self.Q_shape)

    def forward(self, x, targets = None):


        x_old = x.clone()
        x_encod = self.encoder(x_old)
        x = self.NO_Block(x_encod)
        x = self.decoder(x)


        return x
    
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
    
class INO(nn.Module):
    def __init__(self, n_modes, 
                 P_shape = [20,256,32], 
                 Q_shape = [32,256,20],
                   no_skip = 'linear', 
                   conv = 'fourier',
                   level = None,
                    n_ino = 1,
                   nonlinearity=F.gelu):
        
        super(INO, self).__init__()

        self.n_modes = n_modes
        self.Q_shape = Q_shape
        self.P_shape = P_shape
        self.no_skip = no_skip
        self.conv = conv
        self.level = level
        self.n_ino = n_ino
        self.nonlinearity = nonlinearity

        self.hidden_channels = self.P_shape[-1]

        self.encoder = MLP(self.P_shape)
        self.NO_Block = NO_Block(n_modes = self.n_modes, 
                                  channels=self.hidden_channels, 
                                  skip=self.no_skip, 
                                  conv = self.conv,
                                    level = self.level)
        
        self.res = Identity() 
        
        self.decoder = MLP(self.Q_shape)

    def forward(self, x, targets = None):


        x_old = x.clone()
        x = self.encoder(x_old)
        for _ in range(self.n_ino):
            x = self.NO_Block(x)

        x = self.nonlinearity(x)*(1/self.n_ino) 
        x = self.decoder(x) + self.res(x_old)


        return x

class NO_Block(nn.Module):
    def __init__(self, n_modes, channels, skip, nonlinearity=F.gelu, conv = 'fourier',
                 level = None):
        super(NO_Block, self).__init__()
        self.n_modes = n_modes
        self.channels = channels
        self.skip = skip
        self.nonlinearity = nonlinearity
        self.conv = conv
        self.level = level

        if conv == 'fourier':
            self.convs = SpectralConv( self.channels, self.n_modes)
        elif conv == 'wavelet':
            self.convs = WaveConv(self.channels, self.n_modes, level = self.level)

        if self.skip == 'linear':
            self.skip = nn.Conv2d(self.channels, self.channels, kernel_size = (1,1), stride = (1,1))
        elif self.skip == 'identity':
            self.skip = Identity()   

    def forward(self, x):

        x_old = x.clone()



        x_convs = self.convs(x_old)

        if self.skip == 'None':
            x = x_convs
        else:
            x_skip = self.skip(x_old)
            x = x_convs + x_skip

        #nonlinear activation
        # x = self.nonlinearity(x)
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
                 
class SpectralConv(nn.Module):
    def __init__(self,channels,  
                 modes):       
        super(SpectralConv, self).__init__()

        self.in_channels = channels
        self.out_channels = channels
        self.modes = modes

        self.scale = 1 / (self.in_channels * self.out_channels)


        self.weights_im = nn.Parameter(self.scale * torch.rand(channels, channels, self.modes, self.modes) )
        self.weights_re = nn.Parameter(self.scale * torch.rand(channels, channels, self.modes, self.modes) )



    def forward(self, x):

        x_ft = torch.fft.fft2(x, dim = (-2,-1))[...,:self.modes, :self.modes]

        out_ft_re = torch.einsum("bikl,iokl->bokl", x_ft.real, self.weights_re) - torch.einsum("bikl,iokl->bokl", x_ft.imag, self.weights_im )
        out_ft_im = torch.einsum("bikl,iokl->bokl", x_ft.real, self.weights_im) + torch.einsum("bikl,iokl->bokl", x_ft.imag, self.weights_re)
        

        out_ft = out_ft_re + 1j* out_ft_im


        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)), dim = (-2,-1))
        
        return x
    
class WaveConv(nn.Module):
    def __init__(self, channels, 
                modes,
                level = 1, wavelet = 'db1', mode='symmetric'):
        super(WaveConv, self).__init__()

        self.channels = channels

        self.wavelet = wavelet       
        self.mode = mode
        self.level = level


        self.modes = modes
        
        # Parameter initilization
        self.scale = (1 / (channels * channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(channels, channels, self.modes, self.modes))
        self.weights2 = nn.Parameter(self.scale * torch.rand(channels, channels, self.modes, self.modes))
        self.weights3 = nn.Parameter(self.scale * torch.rand(channels, channels, self.modes, self.modes))
        self.weights4 = nn.Parameter(self.scale * torch.rand(channels, channels, self.modes, self.modes))

    def forward(self, x):

        dwt = DWT(J=self.level, mode=self.mode, wave=self.wavelet).to(x.device)
        idwt = IDWT(mode=self.mode, wave=self.wavelet).to(x.device)

        x_ft, x_coeff = dwt(x)

        out_ft = torch.zeros_like(x_ft)
        out_coeff = [torch.zeros_like(coeffs, device= x.device) for coeffs in x_coeff]
        
        out_ft = torch.einsum("bixy,ioxy->boxy", x_ft, self.weights1)
        out_coeff[-1][:,:,0,:,:] = torch.einsum("bixy,ioxy->boxy", x_coeff[-1][:,:,0,:,:].clone(), self.weights2)
        out_coeff[-1][:,:,1,:,:] = torch.einsum("bixy,ioxy->boxy", x_coeff[-1][:,:,1,:,:].clone(), self.weights3)
        out_coeff[-1][:,:,2,:,:] = torch.einsum("bixy,ioxy->boxy", x_coeff[-1][:,:,2,:,:].clone(), self.weights4)
        
        x = idwt((out_ft, out_coeff))
        
        return x

# %%
