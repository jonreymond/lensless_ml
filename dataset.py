import numpy as np
import keras
import os
import glob
import cv2
import tensorflow as tf
from utils import get_shape, rgb2gray


#######################################################################
########################## U-net ###################################### 
#######################################################################

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, dataset_config, indexes, greyscale=False, batch_size=8, use_crop=False, seed=1):
        super().__init__()
        'Initialization'
        # extract filenames
        dir_x = os.path.join(dataset_config['path'], dataset_config['measure_folder'])
        x_filenames = sorted([name for name in os.listdir(dir_x)])

        dir_y =  os.path.join(dataset_config['path'], dataset_config['truth_folder'])
        y_filenames = sorted([name for name in os.listdir(dir_y)])

        assert x_filenames == y_filenames, "the lensed filenames must be equal to the lensless filenames"

        x_filenames = np.asarray([os.path.join(dir_x, name) for name in x_filenames])
        y_filenames = np.asarray([os.path.join(dir_y, name) for name in y_filenames])

        self.in_dim = get_shape(dataset_config, measure=True, greyscale=greyscale)
        self.out_dim = get_shape(dataset_config, measure=False, greyscale=greyscale)
        self.batch_size = batch_size
        self.x_filenames = x_filenames[indexes]
        self.y_filenames = y_filenames[indexes]
        self.num_files = len(indexes)
        self.indexes = np.arange(self.num_files)
        self.seed = seed
        self.greyscale = greyscale

        self.on_epoch_end()


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.num_files / self.batch_size))


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
        X = np.empty((self.batch_size, *self.in_dim), dtype=np.float32)
        Y = np.empty((self.batch_size, *self.out_dim), dtype=np.float32)

        # load data TODO: check if vectorize
        for i, batch_idx in enumerate(batch_indexes):
            # numpy image: H x W x C
            X[i,] = rgb2gray(np.load(self.x_filenames[batch_idx])).transpose((2, 0, 1))
            Y[i,] = rgb2gray(np.load(self.y_filenames[batch_idx])).transpose((2, 0, 1))

        return X, Y
    



#########################################################################
########################## flatnet ###################################### 
#########################################################################
MAX_UINT8_VAL = 2**8 -1
MAX_UINT16_VAL = 2**16 -1

# TODO: Not for test, see implementation for case test
class FlatnetDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, dataset_config, indexes, batch_size=8, use_crop=True, gaussian_noise=0, seed=1):

        super().__init__()

        self.data_conf = dataset_config
        # TODO : maybe define split w.r.t the parent folder and not by files
        x_filenames, y_filenames = self.__get_filenames()
        # TODO : change after // here 4 channels input for input
        self.in_dim = get_shape(dataset_config, measure=True)
        self.out_dim = get_shape(dataset_config, measure=False)
        self.batch_size = batch_size

        self.x_filenames = x_filenames[indexes]
        self.y_filenames = y_filenames[indexes]
        self.num_files = len(indexes)
        self.indexes = np.arange(self.num_files)
        self.seed = seed
        self.gaussian_noise = gaussian_noise
        self.crop = use_crop

        if use_crop:
            self.crop_x_low = dataset_config['crop']['meas_centre_x'] - dataset_config['crop']['meas_crop_size_x'] // 2
            self.crop_x_high = self.crop_x_low + dataset_config['crop']['meas_crop_size_x']
            
            self.crop_y_low = dataset_config['crop']['meas_centre_y'] - dataset_config['crop']['meas_crop_size_y'] // 2
            self.crop_y_high = self.crop_y_low + dataset_config['crop']['meas_crop_size_y']
            
            pad_x = (dataset_config['psf']['height'] - dataset_config['crop']['meas_crop_size_x']) // 2
            pad_y = (dataset_config['psf']['width'] - dataset_config['crop']['meas_crop_size_y']) // 2
            self.padding = [(0, 0), (pad_x, pad_x), (pad_y, pad_y)]
            self.in_dim = self.__get_x(self.x_filenames[0]).shape

        self.on_epoch_end()
 


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.num_files / self.batch_size)) 


    def __getitem__(self, index):
        'Generate one batch of data'
        batch_indexes = self.indexes[index * self.batch_size : (index+1) * self.batch_size]

        X, Y = self.__batch_generation(batch_indexes)
        return X, Y


    def on_epoch_end(self):
        'Updates indexes after each epoch'
        np.random.seed(self.seed)
        np.random.shuffle(self.indexes)


    def __get_filenames(self):
        # if self.data_conf['mode'] == 'test':
        #     raise NotImplementedError

        dir_x = os.path.join(self.data_conf['path'], self.data_conf['measure_folder'])
        dir_y = os.path.join(self.data_conf['path'], self.data_conf['truth_folder'])

        x_filenames = sorted(glob.glob(dir_x + '/*/*'), key=lambda f: os.path.basename(f).replace('.png','').replace('.', ''))
        y_filenames = sorted(glob.glob(dir_y + '/*/*'), key=lambda f: os.path.basename(f).replace('.JPEG',''))
        
        x_names = [os.path.basename(f).replace('.png','').replace('.', '') for f in x_filenames]
        y_names = [os.path.basename(f).replace('.JPEG','') for f in y_filenames]


        assert x_names == y_names, 'some of the samples do not match : list(groundtruths) != list(measurements)'

        return np.asarray(x_filenames), np.asarray(y_filenames)
    

    def __batch_generation(self, batch_indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.in_dim), dtype=np.float32)
        Y = np.empty((self.batch_size, *self.out_dim), dtype=np.float32)

        # load data TODO: check if vectorize
        for i, batch_idx in enumerate(batch_indexes):
            # numpy image: H x W x C
            a = self.__get_x(self.x_filenames[batch_idx])
            b = self.__get_y(self.y_filenames[batch_idx])

            X[i,] = self.__get_x(self.x_filenames[batch_idx])
            Y[i,] = self.__get_y(self.y_filenames[batch_idx])
        return X, Y


    def __get_y(self, filename):
        # read as uint8
        img = cv2.imread(filename)[:, :, ::-1] / MAX_UINT8_VAL
        # TODO : check if width and height in right order
        img = cv2.resize(img, (self.data_conf['truth_width'], self.data_conf['truth_height']))

         # Change range to [-1, 1] range
        img = (img - 0.5) * 2
        img = np.transpose(img, (2, 0, 1))
        return img.astype(np.float32)
        

    def __get_x(self, filename):
        # -1 :return the loaded image as is (with alpha channel, otherwise it gets cropped) 
        # read as uint16
        img = extract_bayer_raw(filename)

        # TODO : try to understand + check if correct since img 4 dim
        if self.crop:
            # Replicate padding
            img = img[self.crop_x_low : self.crop_x_high,
                      self.crop_y_low : self.crop_y_high,]
            img = img.transpose((2, 0, 1))
            img = np.pad(img, self.padding, mode='edge')

        else :
            img = img.transpose((2, 0, 1))

        # Change range to [-1, 1] range
        img = (img - 0.5) * 2
        img += np.random.normal(size=img.shape, scale=self.gaussian_noise)
        return img 

    

def extract_bayer_raw(filename):
    raw = cv2.imread(filename, -1) / MAX_UINT16_VAL

    raw_h, raw_w = raw.shape
    img = np.zeros((raw_h // 2, raw_w // 2, 4), dtype=np.float32)

    img[:, :, 0] = raw[0::2, 0::2]  # r
    img[:, :, 1] = raw[0::2, 1::2]  # gr
    img[:, :, 2] = raw[1::2, 0::2]  # gb
    img[:, :, 3] = raw[1::2, 1::2]  # b
    # tform = skimage.transform.SimilarityTransform(rotation=0.00174)
    # im1=skimage.transform.warp(im1,tform)
    return img
        