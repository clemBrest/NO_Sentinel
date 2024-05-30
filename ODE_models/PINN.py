import torch
from torch import nn,  autograd
from neuralKoopman import KoopmanAE
import torch.nn.functional as F

class Prior(nn.Module):
    def __init__(self, PriorArch = [3,256,1024,256,20], 
                 activation = 'tanh'):
        """
        MLP with a specified architecture, for instance pmodel = [3,256,1024,256,20]
        """
        super(Prior, self).__init__()

        self.PrioArch = PriorArch
        self.activation = activation

        if activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'relu':
            self.activation = F.relu
        elif activation == 'tanh':
            self.activation = F.tanh

        self.layers = nn.ModuleList()
        for i in range(len(PriorArch)-1):
            self.layers.append(nn.Linear(PriorArch[i], PriorArch[i+1]))

    def forward(self, x, y, t):
        # Concatenate the inputs
        inputs = torch.cat((x, y, t), dim=-1)

        # Pass through MLP
        U = inputs
        for i, layer in enumerate(self.layers):
            U = layer(U)
            if i != len(self.layers) - 1:  # Don't apply activation to last layer
                U = torch.gelu(U)

        grad = autograd.grad(
        outputs=U,
        inputs=t,
        grad_outputs=torch.ones_like(U),
        create_graph=True,
        retain_graph=True,
    )[0]

        return x, grad


class PINN(nn.Module):
    def __init__(self, **pmodel):
        """
        PINN class, comprising a Prior as a MLP and Neural ODE model like Koopman AutoEncoder.
        The Neural ODE model inform the Prior about the dynamics, instead of physical laws.
        """

        super(PINN, self).__init__()

        self.model = KoopmanAE(pmodel['input_dim'], pmodel['linear_dims'])
    
        self.Prior = Prior(pmodel['PriorArch'])

    def forward(self, x, y, t, Ut):
        """
        Perform forward pass through the model.

        Args:
            x (torch.Tensor): Input state.

        Returns:
        G(u)(y)
        """

        Utp1, _ = self.KoopmanAE(Ut)
        x, grad = self.Prior(x, y, t,)

        x = 

        return  Utp1, x, grad
    
    def compute_loss_dynamical_model(self, Ut, ):
        """
        Compute the loss of the dynamical model, i.e. the Neural ODE.
        """

        Utp1, x, grad = self(x, y, t, Ut)

        # Compute the loss
        loss = F.mse_loss(Utp1, Ut) + F.mse_loss(grad, Ut)

        return loss



    