#%%
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

from script.NO_Sentinel.datasets.sentinel_data import SentinelDataset
from script.NO_Sentinel.utils.parser import Config
from script.NO_Sentinel.utils.writer import summary_file
from script.NO_Sentinel.lightning.Lmodel import Lmodel

#%%

###################################################################
#       Arguments
###################################################################

args =  Config('configFilterConv.ini')

pmodel = args.model.__dict__


################################################################
#       Model
################################################################

model = Lmodel(model_name = args.model_name,
                lr =  args.learning_rate,
                loss_weights = args.loss_weights,
                future = args.future, 
                **pmodel)

# #train from a checkpoint
# model = RecurentN0.load_from_checkpoint(
#     '/users/local/c23lacro/script/2005/runs/21051056/epoch=640-step=18589.ckpt')


#################################################################
#       Data
################################################################

train_data = SentinelDataset(path_data= args.path_data, 
                             n_train= args.n_train, 
                             size= args.size, 
                             train = True,
                             future= args.future)

train_loader = DataLoader(train_data, batch_size= args.batch_size, 
                          shuffle=True, num_workers=1, pin_memory=True)

test_data = SentinelDataset(path_data= args.path_data,
                            n_train= args.n_train, 
                            size= args.size, 
                            train = False,
                            future= args.future)

test_loader = DataLoader(test_data, batch_size=args.batch_size, 
                         shuffle=False, num_workers=1, pin_memory=True)

###############################################################
#       Summary file
################################################################


summary_file(args, model, train_data, test_data, args.str_name)

################################################################
#       Trainer
################################################################

checkpoint_callback = ModelCheckpoint(
    monitor='eval_loss_total',  # Nom de la métrique à utiliser pour sélectionner les meilleurs modèles
    dirpath=args.saving_path +'/'+ args.str_name,  # Répertoire où enregistrer les modèles
    filename='{epoch}-{val_loss:.2f}',
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