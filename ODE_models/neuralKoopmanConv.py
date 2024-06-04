import torch
import torch.nn as nn
import torch.nn.functional as F


class KoopmanConv(nn.Module):
    def __init__(self, **pmodel):
        """
        Koopman Autoencoder class, comprising an auto-encoder and a Koopman matrix.

        Args:
            input_dim (int): Dimension of the input data.
            linear_dims (list): List of linear layer dimensions.
            device (str, optional): Device to run the model on (default: 'cpu').
        """
        super(KoopmanConv, self).__init__()
    
        self.input_dim = pmodel['input_dim']
        self.linear_dims = pmodel['linear_dims']

        self.latent_dim = self.linear_dims[-1]

        # Encoder layers
        self.encoder = Encoder(self.input_dim, self.linear_dims)

        # Decoder layers
        self.decoder = Decoder(self.input_dim, self.linear_dims)

        self.conv = nn.Conv2d(self.latent_dim,self.latent_dim, kernel_size=pmodel['kernel_size'], stride=1)

    def one_step_ahead(self, x):
        """Predict one-step-ahead in the latent space using the Koopman operator."""
        return self.conv(x)


    def forward(self, x, n):
        """
        Perform forward pass through the model.

        Args:
            x (torch.Tensor): Input state.

        Returns:
            x_advanced (torch.Tensor): Estimated state after one time step.
            phi (torch.Tensor): Encoded input state.
            phi_advanced (torch.Tensor): Encoded input state advanced by one time step.
        """
        # phi = self.encoder(x)
        # phi_advanced = self.one_step_ahead(phi)
        # x_advanced = self.decoder(phi_advanced)
        # return x_advanced, phi, phi_advanced
        return self.forward_n_remember(x,n)

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
    