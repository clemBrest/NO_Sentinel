#%%
from torch.utils.data import DataLoader
import torch
import lightning as L
# from pytorch_lightning.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from sentinel_data import SentinelDataset
from parser import Config
from writer import summary_file
from Lmodel import RecurentN0

#%%

###################################################################
#       Arguments
###################################################################

args =  Config('config.ini')

################################################################
#       Model
################################################################




model = RecurentN0(n_modes=args.n_modes, 
                    P_shape = args.P_shape, 
                    Q_shape = args.Q_shape, 
                    no_skip = args.no_skip,
                    future = args.future,
                    loss_weights=args.loss_weights,
                    conv = args.conv,
                    level = args.level,
                    n_ino = args.n_ino)

# #train from a checkpoint
# model = RecurentN0.load_from_checkpoint(
#     '/users/local/c23lacro/script/2005/runs/21051056/epoch=640-step=18589.ckpt')


#################################################################
#       Data
################################################################

train_data = SentinelDataset(path_data=args.path_data, 
                             n_train=args.n_train, 
                             size=args.size, 
                             train = True,
                             future=args.future)

train_loader = DataLoader(train_data, batch_size=args.batch_size, 
                          shuffle=True, num_workers=1, pin_memory=True)

test_data = SentinelDataset(path_data=args.path_data,
                            n_train=args.n_train,
                            train = False, 
                            size=args.size, 
                            future=args.future)

test_loader = DataLoader(test_data, batch_size=args.batch_size, 
                         shuffle=False, num_workers=1, pin_memory=True)

###############################################################
#       Summary file
################################################################

summary_file(args, model, train_data, test_data)

################################################################
#       Trainer
################################################################
from datetime import datetime
current_time = datetime.now()
time_str = current_time.strftime('%H%M')

checkpoint_callback = ModelCheckpoint(
    monitor='eval_loss_total',  # Nom de la métrique à utiliser pour sélectionner les meilleurs modèles
    dirpath=args.saving_path +'/'+time_str,  # Répertoire où enregistrer les modèles
    filename='{epoch}-{val_loss:.2f}',
    save_top_k=3,  # Nombre de modèles à conserver
    mode='min'  # Mode de sélection des meilleurs modèles ('min' pour minimiser la métrique, 'max' pour la maximiser)
)

logger = TensorBoardLogger(args.saving_path +'/'+ time_str, name=None)


trainer = L.Trainer(max_epochs=args.n_epochs,accelerator="gpu", 
                    devices=[1], logger=logger, 
                    callbacks=[checkpoint_callback],
                    log_every_n_steps=10)

################################################################
#       Training
################################################################

trainer.fit(model, train_loader, test_loader)