import numpy as np
import keras
import os


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, dataset_config, indexes, seed=1):
        'Initialization'
        # extract filenames
        dir_x = os.path.join(dataset_config['path'], dataset_config['lensless'])
        x_filenames = sorted([name for name in os.listdir(dir_x)])

        dir_y =  os.path.join(dataset_config['path'], dataset_config['lens'])
        y_filenames = sorted([name for name in os.listdir(dir_y)])

        assert x_filenames == y_filenames, "the lensed filenames must be equal to the lensless filenames"

        x_filenames = [os.path.join(dir_x, name) for name in x_filenames]
        y_filenames = [os.path.join(dir_y, name) for name in y_filenames]

        self.dim = dataset_config['shape']
        self.batch_size = dataset_config['batch_size']
        self.x_filenames = x_filenames[indexes]
        self.y_filenames = y_filenames[indexes]
        self.num_files = len(indexes)
        self.indexes = np.arange(self.num_files)
        self.seed = seed

        self.on_epoch_end()


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.num_files) / self.batch_size))


    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        batch_indexes = self.indexes[index * self.batch_size : (index+1) * self.batch_size]

        X, Y = self.__batch_generation(batch_indexes)
        return X, Y


    def on_epoch_end(self):
        'Updates indexes after each epoch'
        np.random.seed(self.seed)
        np.random.shuffle(self.indexes)

    def __batch_generation(self, batch_indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        Y = np.empty((self.batch_size, *self.dim))

        # load data TODO: check if vectorize
        for i, batch_idx in enumerate(batch_indexes):
            X[i,] = np.load(self.x_filenames[batch_idx])
            Y[i,] = np.load(self.y_filenames[batch_idx])

        return X, Y