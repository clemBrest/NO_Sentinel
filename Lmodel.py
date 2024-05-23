#%%
import torch
from torch import nn
import torch.nn.functional as F
import lightning as L
from pytorch_lightning.utilities.model_summary import ModelSummary, LayerSummary
import sys
from model import NO, INO
#%%
class RecurentN0(L.LightningModule):
    def __init__(self, n_modes=4, 
                 P_shape=[20,256,32], 
                 Q_shape=[32,256,20], 
                 no_skip='linear', 
                 learning_rate=1e-3,
                 loss_weights = {'reconstruction' : 0.1, 
                                 'linear' : 0.1, 
                                 'ortho_w' : 0.1,
                                 'ortho_conv': 0.1,
                                 'encode' : 0.1,
                                 'exponential' : 0.1},
                future = 35,
                conv = 'fourier',
                level = None,
                n_ino = 1
                ):
        
        super(RecurentN0, self).__init__()

        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.loss_weights = loss_weights
        self.no_skip = no_skip
        self.n_modes = n_modes
        self.P_shape = P_shape
        self.Q_shape = Q_shape
        self.future = future
        self.conv = conv
        self.level = level
        self.n_ino = n_ino

        if self.n_ino == 0:
            self.OneStepNO = NO(n_modes=self.n_modes, 
                    P_shape= self.P_shape, 
                    Q_shape = self.Q_shape,
                    no_skip = self.no_skip,
                    conv = self.conv,
                    level = self.level
                    )
        else:
            self.OneStepNO = INO(n_modes=self.n_modes,
                    P_shape= self.P_shape, 
                    Q_shape = self.Q_shape,
                    no_skip = self.no_skip,
                    conv = self.conv,
                    level = self.level,
                    n_ino = self.n_ino
                    )


    def forward(self, x, future=10, targets = None):
        outputs = []
        inputs = x

        # targets is here for the linearity loss

        for i in range(future):
            if targets is not None:

                out = self.OneStepNO(inputs, targets[:,i,...])
            else:
                out = self.OneStepNO(inputs, targets)

            outputs.append(out)
            # inputs = torch.cat((out, out-inputs), dim=1)
            inputs = out.clone()
        outputs = torch.stack(outputs, dim=1)
        return outputs
    
    def compute_loss(self, inputs, outputs, targets):

        self.losses = {}

        ################################################################
        #       Orthogonality Loss
        ################################################################

        encod_inp = self.OneStepNO.encoder(inputs)

        if 'ortho_w' in self.loss_weights.keys():
            encod_w = self.OneStepNO.NO_Block.skip(encod_inp)

            self.losses['ortho_w'] = self.criterion(encod_inp, encod_w)

        if 'ortho_conv' in self.loss_weights.keys():

            encode_R = self.OneStepNO.NO_Block.convs(encod_inp)

            self.losses['ortho_conv'] = self.criterion(self.criterion(encod_inp, torch.zeros_like(encod_inp)),
                                                        self.criterion(encode_R, torch.zeros_like(encode_R)))

        ################################################################
        #      Encoder-Decoder Loss
        ################################################################

        if 'encode' in self.loss_weights.keys():

            encode_inp = self.OneStepNO.encoder(inputs)
            decode_inp = self.OneStepNO.decoder(encode_inp)

            self.losses['encode'] = self.criterion(inputs, decode_inp)

        ################################################################
        #      Linear Loss
        ################################################################

        if 'linear' in self.loss_weights.keys():

            encode_tar = self.OneStepNO.encoder(targets[:,0,...])
            encode_out = self.OneStepNO.NO_Block(
                     self.OneStepNO.encoder(inputs))

            self.losses['linear'] = self.criterion(encode_tar, encode_out)

        ################################################################
        #      Reconstruction Loss
        ################################################################

        if 'reconstruction' in self.loss_weights.keys():
            if 'exponential' in self.loss_weights.keys():
                self.losses['reconstruction'] = self.criterion_exp(outputs, targets, self.loss_weights['exponential'])
            else:
                self.losses['reconstruction'] = self.criterion(outputs, targets)


        ################################################################
        #      Total Loss
        ################################################################

        self.losses['total'] = sum(self.losses[key] * self.loss_weights[key] for key in self.loss_weights.keys() if (key != 'exponential') )

        self.losses['reconstruction'] = self.criterion(outputs, targets)


    def training_step(self, batch, batch_idx):

        inputs, targets = batch['x'], batch['y']
        outputs = self(inputs, self.future, targets)

        self.compute_loss(inputs, outputs, targets)

        for key in self.losses.keys():
            self.log(f'train_loss_{key}', self.losses[key], on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return self.losses['total']

    def validation_step(self, batch, batch_idx):

        inputs, targets = batch['x'], batch['y']
        outputs = self(inputs, self.future, targets)

        self.compute_loss(inputs, outputs, targets)

        for key in self.losses.keys():
            self.log(f'eval_loss_{key}', self.losses[key], on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return self.losses['total']

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
        return [optimizer], [scheduler]
    
    @staticmethod
    def criterion(x,y):
        return torch.sqrt(torch.mean((x-y)**2))
    
    @staticmethod
    def criterion_exp(x,y,a=0):
        factor = torch.exp(
                    torch.arange(0,x.shape[1])*a
        ).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(x.device)
        return torch.mean(factor*(x-y)**2)
    
    
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = RecurentN0(n_modes=32 , 
                       P_shape = [20,64,8], 
                       Q_shape = [8,64,20], 
                       no_skip = 'linear',
                        future = 100,
                        conv = 'wavelet',
                        learning_rate=1e-3,
                        loss_weights = {'reconstruction' : 0.1, 
                                        'linear' : 0.1, 
                                        'ortho_w' : 0,
                                        'ortho_conv': 0.1,
                                        'encode' : 0.1})


    model = model.to(device)

    print(ModelSummary(model),'\n')
    print(model.OneStepNO)

    #layer summary loop
    for layer in model.OneStepNO.children():
        lSum = LayerSummary(layer)
        print(lSum.layer_type, ': ', lSum.num_parameters,' parameters')
        sys.stdout.flush()

    # print(model.OneStepFNO.encoder)
    sys.stdout.flush()

    #pass a forward pass
    inp = torch.randn(1,20,64,64).to(device)
    tar = torch.randn(1,100,20,64,64).to(device)

    out = model(inp,100)
# %%
