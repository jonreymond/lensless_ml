import hydra
from dataset import DataGenerator
from unet import u_net
import numpy as np
import os
from sklearn.model_selection import train_test_split
import lpips



@hydra.main(version_base=None, config_path="../configs", config_name="ml_reconstruction")
def main(config):
    #blablabla

    # TODO : correct to get directly from spec 
    dataset_config = config['dataset']
    num_files = len([name for name in os.listdir(os.path.join(dataset_config['path'], dataset_config['lensless']))])

    indexes = np.arange(num_files)

    train_indexes, val_indexes = train_test_split(indexes, 
                                                  test_size=dataset_config['validation_split'], 
                                                  shuffle=True,
                                                  random_state=config['seed'])
    
    # Data Generators
    train_generator = DataGenerator(dataset_config, train_indexes, config['seed'])
    val_generator = DataGenerator(dataset_config, val_indexes, config['seed'])

    lpips_loss = None







