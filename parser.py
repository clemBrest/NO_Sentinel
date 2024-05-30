#%%
import configparser

"""
This class is used to parse the configuration file.
It reads the configuration file and stores the parameters in the class attributes.
For loss weights, it only stores the parameters that are not zero, in a dictionary.
"""

class Config:
    def __init__(self, config_file):

        config = configparser.ConfigParser()
        config.read(config_file)
        self.config = config

        self.model = ModelConfig(config)

        self.model_name = config.get('Model', 'model_name')
        self.learning_rate = config.getfloat('Training', 'learning_rate')
        self.future = config.getint('Training', 'future')

        self.remove_None()
        self.set_loss_weights()

        self.saving_path = config.get('Training', 'saving_path')
        self.n_epochs = config.getint('Training', 'n_epochs')
        self.path_data = config.get('Training', 'path_data')
        self.batch_size = config.getint('Training', 'batch_size')
        self.n_train = config.getint('Training', 'n_train')
        self.size = config.getint('Training', 'size')
        self.future = config.getint('Training', 'future')

        self.set_str_name()

    def remove_None(self):
        for attr in self.__dict__:
            if self.__dict__[attr] == None:
                self.__dict__.pop(attr)
    
        
    def set_loss_weights(self):
        self.loss_weights = {}
        config = self.config
        for param in config.options('LossWeights'):
            value = config.getfloat('LossWeights', param)
            if value != 0.0:
                self.loss_weights[param] = value

        if self.model.no_skip == 'identity' and 'ortho_w' in self.loss_weights.keys():
            #remove an element from the dictionary
            self.loss_weights.pop('ortho_w')

    def set_str_name(self):
        if self.model_name == 'NO':
            if 'fourier' in self.model.conv:
                self.str_name = f"{self.model_name}_{self.model.conv}_{self.model.n_modes}_{self.model.no_skip}_Res:{self.model.residual}_lr:{self.learning_rate}_batch:{self.batch_size}"
            if 'wavelet' in self.model.conv:
                self.str_name = f"{self.model_name}_{self.model.conv}_{self.model.level}_{self.model.no_skip}_Res:{self.model.residual}_lr:{self.learning_rate}_batch:{self.batch_size}"
            if 'FilterConvolution' in self.model.conv:
                self.str_name = f"{self.model_name}_conv:{self.model.conv}_kernel:{self.model.kernel}_stride:{self.model.stride}_skip:{self.model.no_skip}_Res:{self.model.residual}_activation:{self.model.activation}lr:{self.learning_rate}_batch:{self.batch_size}"
        elif self.model_name == 'Koopman':
            self.str_name = f"{self.model_name}_{self.model.linear_dims}_{self.learning_rate}batch:{self.batch_size}"




class ModelConfig:
    def __init__(self, config):

        
        self.Q_shape = [int(x) for x in config.get('Model', 'Q_shape').strip('[]').split(', ')]
        self.P_shape = [int(x) for x in config.get('Model', 'P_shape').strip('[]').split(', ')]
        self.no_skip = config.get('Model', 'no_skip', fallback=None)
        self.conv = config.get('Model', 'conv', fallback=None)
        # self.n_ino = config.getint('Model', 'n_ino', fallback=None)
        self.residual = config.getboolean('Model', 'residual', fallback=False)
        self.size = config.getint('Training', 'size')
        self.activation = config.get('Model', 'activation', fallback='Linear')
        


        if 'wavelet' in self.conv:
            self.wavelet()

        elif 'fourier' in self.conv:
            self.n_modes = config.getint('Model', 'n_modes')
        
        elif 'FilterConvolution' in self.conv:
            self.kernel = [int(x) for x in config.get('Model', 'kernel').strip('[]').split(', ')]
            self.stride = config.getint('Model', 'stride')
            self.padding = config.get('Model', 'padding')
            self.padding = 0 if self.padding == '0' else self.padding
        else:
            raise ValueError('Invalid convolution type')


        self.__dict__.pop('size')

        self.remove_None()


    def wavelet(self):

        from pytorch_wavelets import DWT
        import torch

        self.level = int(self.conv.split('_')[-1])
        self.conv = 'wavelet'

        dwt = DWT(wave='db1', J=self.level, mode= 'symmetric')
        dummy_data = torch.randn( 1,1,self.size, self.size ) 
        mode_data, _ = dwt(dummy_data)

        self.n_modes = mode_data.shape[-1]

    def remove_None(self):
        for attr in self.__dict__:
            if self.__dict__[attr] == None:
                self.__dict__.pop(attr)

















# class DataConfig:
#     def __init__(self, config):

#         # Training parameters
#         self.saving_path = config.get('Training', 'saving_path')
#         self.n_epochs = config.getint('Training', 'n_epochs')
#         self.path_data = config.get('Training', 'path_data')
#         self.batch_size = config.getint('Training', 'batch_size')
#         self.n_train = config.getint('Training', 'n_train')
#         self.size = config.getint('Training', 'size')
#         self.future = config.getint('Model', 'future')



# class Config:
#     def __init__(self, config_file):
#         config = configparser.ConfigParser()
#         config.read(config_file)

#         # self.config = config

#         # Training parameters
#         self.saving_path = config.get('Training', 'saving_path')
#         self.n_epochs = config.getint('Training', 'n_epochs')
#         self.path_data = config.get('Training', 'path_data')
#         self.batch_size = config.getint('Training', 'batch_size')
#         self.learning_rate = config.getfloat('Training', 'learning_rate')
#         self.n_train = config.getint('Training', 'n_train')
#         self.size = config.getint('Training', 'size')

#         self.future = config.getint('Model', 'future')
#         self.n_modes = config.getint('Model', 'n_modes')
#         self.Q_shape = [int(x) for x in config.get('Model', 'Q_shape').strip('[]').split(', ')]
#         self.P_shape = [int(x) for x in config.get('Model', 'P_shape').strip('[]').split(', ')]
#         self.no_skip = config.get('Model', 'no_skip')
#         self.conv = config.get('Model', 'conv')
#         self.n_ino = config.getint('Model', 'n_ino')
#         self.residual = config.getboolean('Model', 'residual')

#         if 'wavelet' in self.conv:

#             from pytorch_wavelets import DWT
#             import torch

#             self.level = int(self.conv.split('_')[-1])
#             self.conv = 'wavelet'

#             print('Wavelet level: ', self.level)

#             dwt = DWT(wave='db1', J=self.level, mode= 'symmetric')
#             dummy_data = torch.randn( 1,1,self.size, self.size ) 
#             mode_data, _ = dwt(dummy_data)

#             self.n_modes = mode_data.shape[-1]
#         else:
#             self.level = None

#         print('Model parameters: ')
#         print('n_modes: ', self.n_modes)
#         print('Q_shape: ', self.Q_shape)
#         print('P_shape: ', self.P_shape)
#         print('no_skip: ', self.no_skip)
#         print('conv: ', self.conv)
#         # print('level: ', self.level)


                                                                                     

#         # Loss weights
#         self.loss_weights = {}
#         for param in config.options('LossWeights'):
#             value = config.getfloat('LossWeights', param)
#             if value != 0.0:
#                 self.loss_weights[param] = value

#         if self.no_skip == 'identity' and 'ortho_w' in self.loss_weights.keys():
#             #remove an element from the dictionary
#             self.loss_weights.pop('ortho_w')

if __name__ == '__main__':
    config = Config('config.ini')
    print(config.model.__dict__)
    print(config.data.__dict__)

    # print(config.loss_weights)
    # if not os.path.exists(config.saving_path):
    #     os.makedirs(config.saving_path)


# %%
