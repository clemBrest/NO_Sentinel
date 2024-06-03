#%%
import argparse
from utils.config import Config

from light.Lmodel import Lmodel

from datasets.sentinel_data import SentinelDataset
from torch.utils.data import DataLoader

from utils.writer import summary_file

import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

#%%

###################################################################
#       Arguments
###################################################################
parser = argparse.ArgumentParser(description='Process a config.ini file.')
parser.add_argument('config_file', type=str, help='Path to the config.ini file')

args =  Config(parser.parse_args().config_file)

kwargs = args.__dict__

################################################################
#       Model
################################################################

model = Lmodel(**kwargs)

#################################################################
#       Data
################################################################

train_data = SentinelDataset(**kwargs, train = True)

train_loader = DataLoader(train_data, batch_size= args.batch_size, 
                          shuffle=True, num_workers=1, pin_memory=True)

test_data = SentinelDataset(**kwargs, train = False)

test_loader = DataLoader(test_data, batch_size=args.batch_size, 
                         shuffle=False, num_workers=1, pin_memory=True)

###############################################################
#       Summary file
################################################################


summary_file(args, model, train_data, test_data)

################################################################
#       Trainer
################################################################

checkpoint_callback = ModelCheckpoint(
    monitor='train_loss_total',  # Nom de la métrique à utiliser pour sélectionner les meilleurs modèles
    dirpath=args.saving_path +'/'+ args.str_name,  # Répertoire où enregistrer les modèles
    filename='{epoch}-{eval_loss_total:.2f}',
    save_top_k=3,  # Nombre de modèles à conserver
    mode='min'  # Mode de sélection des meilleurs modèles ('min' pour minimiser la métrique, 'max' pour la maximiser)
)



logger = TensorBoardLogger(args.saving_path +'/'+ args.str_name, name=None)


trainer = L.Trainer(max_epochs=args.n_epochs,accelerator="gpu", 
                    devices=[0], logger=logger, 
                    callbacks=[checkpoint_callback],
                    log_every_n_steps=10)

################################################################
#       Training
################################################################

trainer.fit(model, train_loader, test_loader)