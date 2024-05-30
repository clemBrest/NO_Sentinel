#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from .neuralKoopman import KoopmanAE


class KoopmanDeepOperatorNet(nn.Module):
    """
    Koopman Deep Operator Net class, comprising an auto-encoder and a Koopman operator. and a trunk network.
    Inspired by:
    Lu, L., Jin, P., Pang, G. et al. 
    Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators. 
    Nat Mach Intell 3, 218–229 (2021). https://doi.org/10.1038/s42256-021-00302-5
    """
    def __init__(self, input_dim:int, linear_dims:list, trunk_dims:list):
        """
        Koopman Deep Operator Net class, comprising an auto-encoder and a Koopman operator. and a trunk network.
        """

        super(KoopmanDeepOperatorNet, self).__init__()

        self.input_dim = input_dim
        self.linear_dims = linear_dims
        self.trunk_dims = trunk_dims

        self.KoopmanAE = KoopmanAE(input_dim, linear_dims)
    
        self.Trunk = MLP(shape = trunk_dims)

    def forward(self, x, y):
        """
        Perform forward pass through the model.

        Args:
            x (torch.Tensor): Input state.

        Returns:
        G(u)(y)
        """
        x_advanced, _ = self.KoopmanAE(x)
        y_advanced = self.Trunk(y)

        return  torch.einsum("ptc,bp->b", x_advanced, y_advanced )


class MLP(nn.Module):
    def __init__(self, shape = [2,512,8192,4096], nonlinearity = F.gelu):
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

if __name__ == "__main__":
    model = KoopmanDeepOperatorNet(20, [512, 256, 32], [2, 512, 4096])
    print(model)
    #count nb paramters
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
# %%
