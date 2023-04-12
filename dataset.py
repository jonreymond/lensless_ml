import numpy as np
import keras
import os
import glob
import cv2
import tensorflow as tf
from utils import get_shape, rgb2gray
from abc import ABC, abstractmethod



class DataGenerator(ABC, keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, dataset_config, indexes,  batch_size=8, greyscale=False, seed=1):
        super().__init__()
        'Initialization'
        self.data_conf = dataset_config
        # extract filenames
        x_filenames, y_filenames = self._get_filenames()
        self.x_filenames = x_filenames[indexes]
        self.y_filenames = y_filenames[indexes]

        self.in_dim = get_shape(dataset_config, measure=True, greyscale=greyscale)
        self.out_dim = get_shape(dataset_config, measure=False, greyscale=greyscale)
        self.batch_size = batch_size
        
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

        X = np.empty((self.batch_size, *self.in_dim), dtype=np.float32)
        Y = np.empty((self.batch_size, *self.out_dim), dtype=np.float32)

        # load data TODO: check if vectorize
        for i, batch_idx in enumerate(batch_indexes):

            X[i,] = self._get_x(self.x_filenames[batch_idx])
            Y[i,] = self._get_y(self.y_filenames[batch_idx])

        return X, Y


    def on_epoch_end(self):
        'Updates indexes after each epoch by shuffling it'
        np.random.seed(self.seed)
        np.random.shuffle(self.indexes)


    @abstractmethod
    def _get_filenames(self):
        """returns all the path of the measurements and ground truth samples as 2 lists of strings
        Args: None
        Returns:
            (list[str], list[str]): 2 lists of the path_names of all the pairs : 1st list for the measurements,
                            2nd list for the ground truth, in the same order as the first list to match
        """
        pass
    
    @abstractmethod
    def _get_x(self, filename):
        """returns the preprocessed measurement sample stored at the 'filename' location

        Args:
            filename (str): file path location to the desired measurement sample
        Returns:
            np.array[float] : array of format Channels x measurement_width x measurement_weight, 
                            where the 'Channel' can be the measurement_channels or 1 if greyscale is True
        """
        pass
    
    @abstractmethod
    def _get_y(self, filename):
        """returns the preprocessed groundtruth sample stored at the 'filename' location

        Args:
            filename (str): file path location to the desired groundtruth sample
        Returns:
            np.array[float] : array of format Channels x groundtruth_width x groundtruth_height, 
                            where the 'Channel' can be the groundtruth_channels or 1 if greyscale is True
        """
        pass





###########################################################################
########################## Wallerlab ###################################### 
###########################################################################

class WallerlabGenerator(DataGenerator):
    'Generates data for Keras'
    def __init__(self, dataset_config, indexes, batch_size=8, greyscale=False, seed=1):
        super().__init__(dataset_config, indexes, batch_size, greyscale, seed)
        

    def _get_filenames(self):
        dir_x = os.path.join(self.data_conf['path'], self.data_conf['measure_folder'])
        x_filenames = sorted([name for name in os.listdir(dir_x)])

        dir_y =  os.path.join(self.data_conf['path'], self.data_conf['truth_folder'])
        y_filenames = sorted([name for name in os.listdir(dir_y)])

        assert x_filenames == y_filenames, "the lensed filenames must be equal to the lensless filenames"

        x_filenames = np.asarray([os.path.join(dir_x, name) for name in x_filenames])
        y_filenames = np.asarray([os.path.join(dir_y, name) for name in y_filenames])
        return x_filenames, y_filenames
    

    def _get_x(self, filename):
        x = np.load(filename)
        if self.greyscale:
            x = rgb2gray(x)
        # to channel first
        return x.transpose((2, 0, 1))
    

    def _get_y(self, filename):
        return self._get_x(filename)
    

#########################################################################
########################## Phlatnet ##################################### 
#########################################################################


MAX_UINT8_VAL = 2**8 -1
MAX_UINT16_VAL = 2**16 -1

# TODO: Not for test, see implementation for case test
class PhlatnetDataGenerator(DataGenerator):
    'Generates data for Keras'
    def __init__(self, dataset_config, indexes, batch_size=8, seed=1, use_crop=True, gaussian_noise=0):

        super().__init__(dataset_config, indexes, batch_size=8, seed=1)

        self.gaussian_noise = gaussian_noise
        self.crop = use_crop
        self.use_padding = dataset_config['padding']

        if use_crop:
                # ts: 
        # low_h: 168 
        # high_h: 1448
        # low_w: 261 
        # high_w: 1669
        # # must match the size involved by the cropping
        # size_h: 1280 
        # size_w: 1408 
            self.crop_config_x = dataset_config['crop']['measurements']
            size_h = self.crop_config_x['high_h'] - self.crop_config_x['low_h']
            size_w = self.crop_config_x['high_w'] - self.crop_config_x['low_w']

            assert size_h == self.crop_config_x['size_h'], 'not same height size from cropping and declared, got ' +str(size_h)+' instead of ' +str(self.crop_config_x['size_h'])
            assert size_w == self.crop_config_x['size_w'], 'not same width size from cropping and declared, got ' +str(size_w)+' instead of ' +str(self.crop_config_x['size_w']) 

        if self.use_padding:
            pad_x = (dataset_config['psf']['height'] - self.crop_config_x['size_h']) // 2
            pad_y = (dataset_config['psf']['width'] - self.crop_config_x['size_w']) // 2
            self.padding = [(pad_x, pad_x), (pad_y, pad_y), (0, 0)]

        self.in_dim = self._get_x(self.x_filenames[0]).shape
        print('in dim: ', self.in_dim)

 


    def _get_filenames(self):
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
    


    def _get_y(self, filename):
        # read as uint8
        img = cv2.imread(filename)[:, :, ::-1] / MAX_UINT8_VAL
        # TODO : check if width and height in right order
        img = cv2.resize(img, (self.data_conf['truth_width'], self.data_conf['truth_height']))

         # Change range to [-1, 1] range
        img = (img - 0.5) * 2
        img = np.transpose(img, (2, 0, 1))
        return img.astype(np.float32)
        

    def _get_x(self, filename):
        # -1 :return the loaded image as is (with alpha channel, otherwise it gets cropped) 
        # read as uint16, channel last
        img = extract_bayer_raw(filename)


        if self.crop:
            # Replicate padding
            img = img[self.crop_config_x['low_h'] : self.crop_config_x['high_h'],
                      self.crop_config_x['low_w'] : self.crop_config_x['high_w'],:]

        if self.use_padding:
            img = np.pad(img, self.padding, mode='edge')

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
        


#########################################################################
########################### Flatnet ##################################### 
#########################################################################


# MAX_UINT8_VAL = 2**8 -1
# MAX_UINT16_VAL = 2**16 -1

# # TODO: Not for test, see implementation for case test
# class FlatnetDataGenerator(DataGenerator):
#     'Generates data for Keras'
#     def __init__(self, dataset_config, indexes, batch_size=8, seed=1, use_crop=True, gaussian_noise=0):

#         super().__init__(dataset_config, indexes, batch_size=8, seed=1)

#         self.gaussian_noise = gaussian_noise
#         self.crop = use_crop
#         # CHECK if same
#         if use_crop:
#             self.crop_x_low = dataset_config['crop']['meas_centre_x'] - dataset_config['crop']['meas_crop_size_x'] // 2
#             self.crop_x_high = self.crop_x_low + dataset_config['crop']['meas_crop_size_x']
            
#             self.crop_y_low = dataset_config['crop']['meas_centre_y'] - dataset_config['crop']['meas_crop_size_y'] // 2
#             self.crop_y_high = self.crop_y_low + dataset_config['crop']['meas_crop_size_y']
            
#             pad_x = (dataset_config['psf']['height'] - dataset_config['crop']['meas_crop_size_x']) // 2
#             pad_y = (dataset_config['psf']['width'] - dataset_config['crop']['meas_crop_size_y']) // 2
#             self.padding = [(0, 0), (pad_x, pad_x), (pad_y, pad_y)]
#             self.in_dim = self._get_x(self.x_filenames[0]).shape
#             print('in dim: ', self.in_dim)

 

#     # Done : SAME
#     def _get_filenames(self):
#         # if self.data_conf['mode'] == 'test':
#         #     raise NotImplementedError

#         dir_x = os.path.join(self.data_conf['path'], self.data_conf['measure_folder'])
#         dir_y = os.path.join(self.data_conf['path'], self.data_conf['truth_folder'])

#         x_filenames = sorted(glob.glob(dir_x + '/*/*'), key=lambda f: os.path.basename(f).replace('.png','').replace('.', ''))
#         y_filenames = sorted(glob.glob(dir_y + '/*/*'), key=lambda f: os.path.basename(f).replace('.JPEG',''))
        
#         x_names = [os.path.basename(f).replace('.png','').replace('.', '') for f in x_filenames]
#         y_names = [os.path.basename(f).replace('.JPEG','') for f in y_filenames]


#         assert x_names == y_names, 'some of the samples do not match : list(groundtruths) != list(measurements)'

#         return np.asarray(x_filenames), np.asarray(y_filenames)
    


#     def _get_y(self, filename):
#         # read as uint8
#         # check if really uint8
#         img = cv2.imread(filename)[:, :, ::-1] / MAX_UINT8_VAL
#         # TODO : check if width and height in right order
#         img = cv2.resize(img, (self.data_conf['truth_width'], self.data_conf['truth_height']))

#          # Change range to [-1, 1] range
#         img = (img - 0.5) * 2
#         img = np.transpose(img, (2, 0, 1))
#         return img.astype(np.float32)
        

#     def _get_x(self, filename):
#         # -1 :return the loaded image as is (with alpha channel, otherwise it gets cropped) 
#         # read as uint16
#         img = extract_bayer_raw(filename)

#         # TODO : try to understand + check if correct since img 4 dim
#         if self.crop:
#             # Replicate padding
#             img = img[self.crop_x_low : self.crop_x_high,
#                       self.crop_y_low : self.crop_y_high,]

#             img = img.transpose((2, 0, 1))
#             img = np.pad(img, self.padding, mode='edge')
#         else :
#             img = img.transpose((2, 0, 1))

#         # Change range to [-1, 1] range
#         img = (img - 0.5) * 2
#         img += np.random.normal(size=img.shape, scale=self.gaussian_noise)
#         return img 

    
# def extract_bayer_raw(filename):
#     raw = cv2.imread(filename, -1) / MAX_UINT16_VAL

#     raw_h, raw_w = raw.shape
#     img = np.zeros((raw_h // 2, raw_w // 2, 4), dtype=np.float32)

#     img[:, :, 0] = raw[0::2, 0::2]  # r
#     img[:, :, 1] = raw[0::2, 1::2]  # gr
#     img[:, :, 2] = raw[1::2, 0::2]  # gb
#     img[:, :, 3] = raw[1::2, 1::2]  # b
#     # tform = skimage.transform.SimilarityTransform(rotation=0.00174)
#     # im1=skimage.transform.warp(im1,tform)
#     return img