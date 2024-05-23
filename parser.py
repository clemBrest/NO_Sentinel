
import configparser
import os

"""
This class is used to parse the configuration file.
It reads the configuration file and stores the parameters in the class attributes.
For loss weights, it only stores the parameters that are not zero, in a dictionary.
In order to avoid overwriting existing files, it checks if the saving path already exists.
"""


class Config:
    def __init__(self, config_file):
        config = configparser.ConfigParser()
        config.read(config_file)

        # Training parameters
        self.saving_path = config.get('Training', 'saving_path')
        self.n_epochs = config.getint('Training', 'n_epochs')
        self.path_data = config.get('Training', 'path_data')
        self.batch_size = config.getint('Training', 'batch_size')
        self.learning_rate = config.getfloat('Training', 'learning_rate')
        self.n_train = config.getint('Training', 'n_train')
        self.size = config.getint('Training', 'size')

        self.future = config.getint('Model', 'future')
        self.n_modes = config.getint('Model', 'n_modes')
        self.Q_shape = [int(x) for x in config.get('Model', 'Q_shape').strip('[]').split(', ')]
        self.P_shape = [int(x) for x in config.get('Model', 'P_shape').strip('[]').split(', ')]
        self.no_skip = config.get('Model', 'no_skip')
        self.conv = config.get('Model', 'conv')
        self.n_ino = config.getint('Model', 'n_ino')

        if 'wavelet' in self.conv:

            from pytorch_wavelets import DWT
            import torch

            self.level = int(self.conv.split('_')[-1])
            self.conv = 'wavelet'

            print('Wavelet level: ', self.level)

            dwt = DWT(wave='db1', J=self.level, mode= 'symmetric')
            dummy_data = torch.randn( 1,1,self.size, self.size ) 
            mode_data, _ = dwt(dummy_data)

            self.n_modes = mode_data.shape[-1]
        else:
            self.level = None

        print('Model parameters: ')
        print('n_modes: ', self.n_modes)
        print('Q_shape: ', self.Q_shape)
        print('P_shape: ', self.P_shape)
        print('no_skip: ', self.no_skip)
        print('conv: ', self.conv)
        # print('level: ', self.level)


                                                                                     

        # Loss weights
        self.loss_weights = {}
        for param in config.options('LossWeights'):
            value = config.getfloat('LossWeights', param)
            if value != 0.0:
                self.loss_weights[param] = value

        if self.no_skip == 'identity' and 'ortho_w' in self.loss_weights.keys():
            #remove an element from the dictionary
            self.loss_weights.pop('ortho_w')

