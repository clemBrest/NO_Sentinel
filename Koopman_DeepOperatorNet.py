#%%
import torch
import torch.nn as nn
import torch.nn.functional as F


class KoopmanAE(nn.Module):
    def __init__(self, input_dim:int, linear_dims:list):
        """
        Koopman Autoencoder class, comprising an auto-encoder and a Koopman matrix.

        Args:
            input_dim (int): Dimension of the input data.
            linear_dims (list): List of linear layer dimensions.
            device (str, optional): Device to run the model on (default: 'cpu').
        """
        super(KoopmanAE, self).__init__()
    
        self.input_dim = input_dim
        self.linear_dims = linear_dims

        self.latent_dim = linear_dims[-1]

        # Encoder layers
        self.encoder = Encoder(input_dim, linear_dims)

        # Decoder layers
        self.decoder = Decoder(input_dim, linear_dims)

        # Koopman operator
        self.K = torch.nn.Parameter(
            torch.eye(self.latent_dim, requires_grad=True)
        )
        self.state_dict()['K'] = self.K
        self.first_call = True

    def one_step_ahead(self, x):
        """Predict one-step-ahead in the latent space using the Koopman operator."""
        return torch.matmul(x, self.K)

    def one_step_back(self, x):
        """Predict one-step-back in the latent space using the inverse of the Koopman operator."""
        return torch.matmul(x, torch.inverse(self.K))

    def forward(self, x):
        """
        Perform forward pass through the model.

        Args:
            x (torch.Tensor): Input state.

        Returns:
            x_advanced (torch.Tensor): Estimated state after one time step.
            phi (torch.Tensor): Encoded input state.
            phi_advanced (torch.Tensor): Encoded input state advanced by one time step.
        """
        phi = self.encoder(x)
        phi_advanced = self.one_step_ahead(phi)
        x_advanced = self.decoder(phi_advanced)
        return x_advanced, phi, phi_advanced

    def forward_n(self, x, n):
        """
        Perform forward pass for n steps.

        Args:
            x (torch.Tensor): Input state.
            n (int): Number of steps to advance.

        Returns:
            x_advanced (torch.Tensor): Estimated state after n time steps.
            phi (torch.Tensor): Encoded input state.
            phi_advanced (torch.Tensor): Encoded state advanced by n time steps.
        """
        phi = self.encoder(x)
        phi_advanced = self.one_step_ahead(phi)
        for k in range(n-1):
            phi_advanced = self.one_step_ahead(phi_advanced)
        x_advanced = self.decoder(phi_advanced)
        return x_advanced, phi, phi_advanced

    def forward_n_remember(self, x, n):
        """
        Perform forward pass for n steps while remembering intermediate latent states.

        Args:
            x (torch.Tensor): Input state.
            n (int): Number of steps to advance.
            training (bool, optional): Flag to indicate training mode (default: False).

        Returns:
            x_advanced (torch.Tensor or None): Estimated state after n time steps if not training, otherwise None.
            phis (torch.Tensor): Encoded state at each step, concatenated along the 0th dimension.
        """

        phis = []
        x_advanceds = []
        phis.append(self.encoder(x))
        for k in range(n):
            phis.append(self.one_step_ahead(phis[-1]))
            x_advanceds.append(self.decoder(phis[-1]))

        return torch.stack(x_advanceds, dim=1), torch.stack(phis, dim=1)

    def backward(self, x):
        """
        Perform backward pass through the model.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            x_advanced (torch.Tensor): Estimated state after one step back.
            phi (torch.Tensor): Encoded input state.
            phi_advanced (torch.Tensor): Encoded state advanced one step back.
        """
        phi = self.encoder(x)
        phi_advanced = self.one_step_back(phi)
        x_advanced = self.decoder(phi_advanced)
        return x_advanced, phi, phi_advanced

    def backward_n(self, x, n):
        """
        Perform backward pass for n steps.

        Args:
            x (torch.Tensor): Input state.
            n (int): Number of steps to advance.

        Returns:
            x_advanced (torch.Tensor): Estimated state after n steps back.
            phi (torch.Tensor): Encoded input state.
            phi_advanced (torch.Tensor): Encoded state advanced n steps back.
        """
        phi = self.encoder(x)
        phi_advanced = self.one_step_back(phi)
        for k in range(n-1):
            phi_advanced = self.one_step_back(phi_advanced)
        x_advanced = self.decoder(phi_advanced)
        return x_advanced, phi, phi_advanced

    def backward_n_remember(self, x, n):
        """
        Perform backward pass for n steps while remembering intermediate states.

        Args:
            x (torch.Tensor): Input state.
            n (int): Number of steps to advance.

        Returns:
            x_advanced (torch.Tensor): Reconstructed state after n steps back.
            phis (torch.Tensor): Encoded state at each step, concatenated along the 0th dimension.
        """
        phis = []
        phis.append(self.encoder(x))
        for k in range(n):
            phis.append(self.one_step_back(phis[-1]))
        x_advanced = self.decoder(phis[n])
        return x_advanced, torch.cat(tuple(phi.unsqueeze(0) for phi in phis), dim=0)

class KoopmanOperator(nn.Module):
    def __init__(self, latent_dim:int):
        """
        Koopman operator class.

        Args:
            latent_dim (int): Dimension of the latent space.
        """
        super(KoopmanOperator, self).__init__()
        self.K = torch.eye(latent_dim, requires_grad=True)

    def forward(self, x):
        """
        Perform forward pass through the Koopman operator.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Transformed data.
        """
        return torch.matmul(x, self.K)

class Encoder(nn.Module):
    def __init__(self, input_dim:int, linear_dims:list):
        """
        Encoder class, comprising a series of linear layers.

        Args:
            input_dim (int): Dimension of the input data.
            linear_dims (list): List of linear layer dimensions.
        """
        super(Encoder, self).__init__()

        self.encoder = nn.ModuleList()
        self.encoder.add_module("encoder_1", nn.Linear(input_dim, linear_dims[0]))
        for i in range(len(linear_dims)-1):
            self.encoder.add_module(f"encoder_{i+2}", nn.Linear(linear_dims[i], linear_dims[i+1]))

    def forward(self, x):
        """
        Perform forward pass through the encoder.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Encoded data.
        """
        for layer_idx, layer in enumerate(self.encoder):
            x = layer(x)
            if layer_idx < len(self.encoder) - 1:
                x = F.relu(x)
        return x
    
class Decoder(nn.Module):
    """
    Decoder class, comprising a series of linear layers.
    """
    def __init__(self, input_dim:int, linear_dims:list):
        """
        Decoder class, comprising
        Args:
            linear_dims (list): List of linear layer dimensions.
        """
        super(Decoder, self).__init__()

        self.decoder = nn.ModuleList()
        for i in range(len(linear_dims)-1):
            self.decoder.add_module(f"decoder_{i+1}", nn.Linear(linear_dims[-i-1], linear_dims[-i-2]))
        self.decoder.add_module(f"decoder_{len(linear_dims)}", nn.Linear(linear_dims[0], input_dim))

    def forward(self, x):
        """
        Perform forward pass through the decoder.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Decoded data.
        """
        for layer_idx, layer in enumerate(self.decoder):
            x = layer(x)
            if layer_idx < len(self.decoder) - 1:
                x = F.relu(x)
        return x
    
class KoopmanDeepOperatorNet(nn.Module):
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
