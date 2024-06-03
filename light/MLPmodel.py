
import torch
import lightning as L
from ODE_models.PINN import Prior

class MLP_model(L.LightningModule):
    def __init__(self, **kwargs):
        
        super(MLP_model, self).__init__()

        self.save_hyperparameters()

        self.learning_rate = kwargs['lr']
        self.kwargs = kwargs

        self.model = Prior( **kwargs )

    def forward(self, X ):
        return self.model(X)
    
    def training_step(self, batch, batch_idx):

        X, tar = batch['X'], batch['tar']
        out = self(X)

        self.loss = self.criterion(out, tar)

        self.log('train_loss_total', self.loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return self.loss
    
    def validation_step(self, batch, batch_idx):

        X, tar = batch['X'], batch['tar']
        out = self(X)

        self.loss = self.criterion(out, tar)

        self.log('eval_loss_total', self.loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return self.loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return [optimizer]
    
    @staticmethod
    def criterion(x,y):
        return torch.mean((x-y)**2)