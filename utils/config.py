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

        for section in config.sections():
            for key in config[section]:
                argument = config[section][key]

                type_arg = self.check_type(argument)

                match type_arg:
                    case 'int':
                        setattr(self, key, config.getint(section, key))
                    case 'float':
                        setattr(self, key, config.getfloat(section, key))
                    case 'bool':
                        setattr(self, key, config.getboolean(section, key))
                    case 'list':
                        item = config.get(section, key).strip('[]').split(', ')
                        setattr(self, key, 
                                [int(x) for x in item]
                                )
                    case 'str':
                        setattr(self, key, config.get(section, key))
                    case 'dict':
                        item = config.get(section, key).strip('{}').split(', ')
                        setattr(self, key,
                                {str(x.split(':')[0]).strip('\"\"'): float(x.split(':')[1]) for x in item}
                                )
                    case _:
                        print('key:', key)
                        raise ValueError (f"Warning: {key} is not a valid type") 

        
        if 'conv' in self.__dict__.keys():
            if 'wavelet' in self.conv:
                self.set_wavelet(config)

        del config

        self.set_str_name()

    def set_wavelet(self, config):

        from pytorch_wavelets import DWT
        import torch

        size = config.getint('Training', 'size')

        dwt = DWT(wave='db1', J=self.level, mode= 'symmetric')
        dummy_data = torch.randn( 1,1,size, size ) 
        mode_data, _ = dwt(dummy_data)

        self.n_modes = mode_data.shape[-1]

    def set_str_name(self):
        str_name = ''
        for key in self.__dict__.keys():
            if key != 'future' and key != 'saving_path' and key != 'path_data' and key != 'loss_weights':
                str_name += f"{key}:{self.__dict__[key]}_"
        str_name = str_name[:-1]
        self.str_name = str_name

    @staticmethod
    def check_type(key):
        try:
            int(key)
            return 'int'
        except:
            pass
        try:
            float(key)
            return 'float'
        except:
            pass
        if key == 'True' or key == 'False':
            return 'bool'
        elif '{' in key:
            return 'dict'
        elif '[' in key:
            return 'list'
        else :
            return 'str'
        
#%%

if __name__ == '__main__':
    args =  Config('/users/local/c23lacro/script/NO_Sentinel/utils/configtest.ini')
    print(args.__dict__)
# %%
