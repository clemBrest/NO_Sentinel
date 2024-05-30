import torch
from torch import nn
from pytorch_wavelets import DWT, IDWT 

class SpectralConv(nn.Module):
    """
    Spectral Convolution Layer using Fourier Transform, with learnable weights.
    Original code from:
    (https://github.com/neuraloperator/neuraloperator)
    """

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
    """
    Wavelet Convolution Layer using Wavelet Transform, with learnable weights.
    Original code from:
    (https://github.com/TapasTripura/Wavelet-Neural-Operator-for-pdes)
    """

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