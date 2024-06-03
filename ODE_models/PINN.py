import torch
from torch import nn,  autograd
from .neuralKoopman import KoopmanAE
import torch.nn.functional as F
# import lightning as L
# import light.Lmodel as Lmodel


class Prior(nn.Module):
    def __init__(self, **kwargs):
        """
        MLP with a specified architecture, for instance pmodel = [3,256,1024,256,20]
        """
        super(Prior, self).__init__()

        self.PriorArch = kwargs['priorarch']

        if kwargs['activation'] == 'gelu':
            self.activation = F.gelu
        elif kwargs['activation'] == 'relu':
            self.activation = F.relu
        elif kwargs['activation'] == 'tanh':
            self.activation = F.tanh

        self.layers = nn.ModuleList()
        for i in range(len(self.PriorArch)-1):
            self.layers.append(nn.Linear(self.PriorArch[i], self.PriorArch[i+1]))

    def forward(self, X):
        # Pass through MLP
        U = X
        for i, layer in enumerate(self.layers):
            U = layer(U)
            if i != len(self.layers) - 1:  # Don't apply activation to last layer
                U = self.activation(U)

        return U

# class PINN(L.LightningModule):
#     def __init__(self, **pmodel):
#         """
#         PINN class, comprising a Prior as a MLP and Neural ODE model like Koopman AutoEncoder.
#         The Neural ODE model inform the Prior about the dynamics, instead of physical laws.
#         """

#         super(PINN, self).__init__()

#         self.future = pmodel['future']
#         self.model = Lmodel.Lmodel(model_name = 'KoopmanAE', **pmodel)
#         self.model = KoopmanAE(pmodel['input_dim'], pmodel['linear_dims'])
    
#         self.Prior = Prior(pmodel['PriorArch'])


    
#     def training_step(self, batch, batch_idx):
#         """
#         Training step for the PINN model.
#         """


#         self.model.train()

#         _ , inp, _ = batch['X'], batch['inp'], batch['tar']

#         x_hat, phi_hat = self.model(inp, self.future)

#         self.model.compute_loss( x_hat, phi_hat, batch )

#         # compute the gradients
#         self.model.optimizer.zero_grad()
#         self.model.losses['total'].backward()
#         self.model.optimizer.step()


#         for key in self.model.losses.keys():
#             self.log(f'train_loss_{key}', self.model.losses[key], on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

#         self.Prior.train()

#         self.compute_pinn_loss(batch)

#         for key in self.loss.keys():
#             self.log(f'train_loss_{key}', self.loss[key], on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

#         self.optimizer.zero_grad()
#         self.loss['total'].backward()
#         self.optimizer.step()

#     def validation_step(self, batch, batch_idx):
#         """
#         Validation step for the PINN model.
#         """

#         self.model.eval()

#         _, inp, _ = batch['X'], batch['inp'], batch['tar']

#         x_hat, phi_hat = self.model(inp, self.future)

#         self.model.compute_loss( x_hat, phi_hat, batch )

#         for key in self.model.losses.keys():
#             self.log(f'eval_loss_{key}', self.model.losses[key], on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

#         self.Prior.eval()

#         self.compute_pinn_loss(batch)

#         for key in self.loss.keys():
#             self.log(f'eval_loss_{key}', self.loss[key], on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    
#     def compute_pinn_loss(self, batch):
#         """
#         Compute the loss for the PINN model.
#         """

#         (x, y, t), U = batch['X'], batch['inp']

#         U_Prior = self.Prior(x, y, t)


#         z_prior = self.model.encoder(U_Prior)
#         z = self.model.encoder(U)

#         dz_priordt = autograd.grad(outputs=z_prior,inputs=t,grad_outputs=torch.ones_like(z_prior),create_graph=True,retain_graph=True )[0]
#         dz_modeldt = autograd.grad(outputs=z,inputs=t,grad_outputs=torch.ones_like(z),create_graph=True,retain_graph=True )[0]

#         self.loss = {}

#         self.loss['reconstruction'] = self.model.criterion(U, U_Prior)
#         self.loss['prior'] = self.model.criterion(dz_priordt, dz_modeldt)

#         self.loss['total'] = self.loss['reconstruction'] + self.loss['closure']

    
#     def configure_optimizers(self):
#         """
#         Configure the optimizers for the PINN model.
#         """
#         self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model.lr)
    