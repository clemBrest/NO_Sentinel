#%%
import torch
import lightning as L
from pytorch_lightning.utilities.model_summary import ModelSummary, LayerSummary
import sys
from PDE_models.model import NO
from ODE_models.Koopman_DeepOperatorNet import KoopmanAE, KoopmanDeepOperatorNet
from ODE_models.neuralKoopmanConv import KoopmanConv

class NO_model(L.LightningModule):
    def __init__(self, **kwargs):
        
        super(NO_model, self).__init__()

        self.save_hyperparameters()
        self.model_name = kwargs['model_name']


        self.set_model(**kwargs )

        self.learning_rate = kwargs['lr']
        self.loss_weights = kwargs['loss_weights']
        self.future =  kwargs['future']
        self.kwargs = kwargs

    def set_model(self, **kwargs):
        match self.model_name:
            case 'NO':
                self.model = NO(**kwargs)
            case 'Koopman':
                self.model = KoopmanAE(**kwargs)
            case 'KoopmanDeepOperatorNet':
                self.model = KoopmanDeepOperatorNet(**kwargs)
            case 'LinConv':
                self.model = KoopmanConv(**kwargs)
            case _:
                raise ValueError(f'Model {self.model_name} not recognized')

    def forward(self, x, future ):
        return self.model(x, future)
    
    
    def compute_loss(self, x_hat, phi_hat, batch):


        inp, tar = batch['inp'], batch['tar']

        self.losses = {}
        

        ################################################################
        #       Orthogonality Loss
        ################################################################

        encod_inp = self.model.encoder(inp)

        if 'ortho_w' in self.loss_weights.keys():
            encod_w = self.model.one_step.skip(encod_inp)

            self.losses['ortho_w'] = self.criterion(encod_inp, encod_w)

        if 'ortho_conv' in self.loss_weights.keys():

            encode_R = self.model.one_step.convs(encod_inp)

            self.losses['ortho_conv'] = self.criterion(self.criterion(encod_inp, torch.zeros_like(encod_inp)),
                                                        self.criterion(encode_R, torch.zeros_like(encode_R)))
            

        ################################################################
        #      Encoder-Decoder Loss
        ################################################################

        if 'encode' in self.loss_weights.keys():

            code_decode_inp = self.model.decoder(phi_hat[:,0,:])
            self.losses['encode'] = self.criterion(inp, code_decode_inp)

        ################################################################
        #      Linear Loss
        ################################################################

        if 'linear' in self.loss_weights.keys():
            
            tar_reshaped = tar.reshape(-1, *tar.shape[2:])

            encode_tar = self.model.encoder(tar_reshaped)

            encoded_tar_reshaped = encode_tar.reshape(tar.shape[0], tar.shape[1], -1, tar.shape[-2], tar.shape[-1])

            self.losses['linear'] = self.criterion(encoded_tar_reshaped, phi_hat[:,1:,:])
   

        ################################################################
        #      Reconstruction Loss
        ################################################################

        if 'reconstruction' in self.loss_weights.keys():
            if 'exponential' in self.loss_weights.keys():
                self.losses['reconstruction'] = self.criterion_exp(x_hat, tar, self.loss_weights['exponential'])
            else:
                self.losses['reconstruction'] = self.criterion(x_hat, tar)


        ################################################################
        #      Total Loss
        ################################################################

        self.losses['total'] = sum(self.losses[key] * self.loss_weights[key] for key in self.loss_weights.keys() if (key != 'exponential') )

        self.losses['reconstruction'] = self.criterion(x_hat, tar)


    def training_step(self, batch, batch_idx):

        inp = batch['inp']
        # outputs = self(inputs, self.future, targets)
        x_hat, phi_hat = self(inp, self.future)

        self.compute_loss( x_hat, phi_hat, batch )

        for key in self.losses.keys():
            self.log(f'train_loss_{key}', self.losses[key], on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return self.losses['total']
    

    def validation_step(self, batch, batch_idx):

        inp = batch['inp']
        # outputs = self(inputs, self.future, targets)
        x_hat, phi_hat = self(inp, self.future)

        self.compute_loss( x_hat, phi_hat, batch)

        for key in self.losses.keys():
            self.log(f'eval_loss_{key}', self.losses[key], on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return self.losses['total']
    

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return [optimizer]
    
    @staticmethod
    def criterion(x,y):
        # return torch.sqrt(torch.mean((x-y)**2))
        return torch.mean((x-y)**2)
    
    @staticmethod
    def criterion_exp(x,y,a=0):
        factor = torch.exp(
                    torch.arange(0,x.shape[1])*a
        ).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(x.device)
        return torch.mean(factor*(x-y)**2)
    










    
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Lmodel(n_modes=32 , 
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
