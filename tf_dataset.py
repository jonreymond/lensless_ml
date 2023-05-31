
from abc import ABC, abstractmethod
from utils import *
from tensorflow.data import Dataset
import tensorflow as tf
import glob

MAX_UINT8_VAL = 2**8 -1

class DataLoader(ABC):
    'Generates data for Keras'
    def __init__(self, dataset_config, indexes, batch_size, greyscale=False, seed=1):
        super().__init__()
        'Initialization'
        self.data_conf = dataset_config
        # extract filenames
        x_filenames, y_filenames = self._get_filenames()


        self.x_filenames = x_filenames[indexes]
        self.y_filenames = y_filenames[indexes]

        self.in_dim = get_shape(dataset_config, measure=True, greyscale=greyscale)
        self.out_dim = get_shape(dataset_config, measure=False, greyscale=greyscale)
        
        self.num_files = len(indexes)
        self.indexes = np.arange(self.num_files)
        self.seed = seed
        self.greyscale = greyscale
        self.batch_size = batch_size


    def get(self):
        # load measurements
        # def map_func(feature_path):
        #     feature = np.load(feature_path)
        #     return feature
        data_measure = Dataset.from_tensor_slices(self.x_filenames).cache()
        data_measure = self._map_x(data_measure)
        # data_measure = data_measure.interleave(self._map_x,
        #                                        cycle_length=36, num_parallel_calls=tf.data.AUTOTUNE,
        #                                        deterministic=False)

        

        data_truth = Dataset.from_tensor_slices(self.y_filenames).cache()
        data_truth = self._map_y(data_truth)
        # .map(lambda item: tf.numpy_function(np.load, [item], tf.float32))
        # load ground truth
        # data_truth = Dataset.from_tensor_slices(self.y_filenames).map(lambda item: tf.numpy_function(
        #   np.load, [item], tf.float32))
        # .map(lambda y: tf.py_function(self._get_y, [y], tf.float32))
        # zip together
        data = Dataset.zip((data_measure, data_truth))
        # shuffle
        data = data.shuffle(buffer_size=1000, reshuffle_each_iteration=True, seed=self.seed)

        # data = data.shuffle(buffer_size=self.num_files, seed=self.seed)
        # batch
        data = data.batch(self.batch_size)
        # prefetch
        data = data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return data


    def get_samples(self, num_samples, shuffle=True):
        """Return num_samples pairs

        Args:
            num_samples (int): number of samples desired

        Returns:
            (numpy array, numpy array): pair of samples
        """
        indexes = np.arange(self.num_files)
        if shuffle:
            np.random.seed(self.seed)
            np.random.shuffle(indexes)
        sample_indexes = indexes[:num_samples]

        X = np.empty((num_samples, *self.in_dim), dtype=np.float32)
        Y = np.empty((num_samples, *self.out_dim), dtype=np.float32)

        for i, batch_idx in enumerate(sample_indexes):
            print(self.x_filenames[batch_idx])
            X[i,] = self._get_x(self.x_filenames[batch_idx])
            Y[i,] = self._get_y(self.y_filenames[batch_idx])

        if num_samples == 1:
            X = X[0,]
            Y = Y[0,]
        return X, Y


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
    def _map_x(self, filename):
        """returns the preprocessed measurement sample stored at the 'filename' location

        Args:
            filename (str): file path location to the desired measurement sample
        Returns:
            np.array[float] : array of format Channels x measurement_width x measurement_weight, 
                            where the 'Channel' can be the measurement_channels or 1 if greyscale is True
        """
        pass
    
    @abstractmethod
    def _map_y(self, filename):
        """returns the preprocessed groundtruth sample stored at the 'filename' location

        Args:
            filename (str): file path location to the desired groundtruth sample
        Returns:
            np.array[float] : array of format Channels x groundtruth_width x groundtruth_height, 
                            where the 'Channel' can be the groundtruth_channels or 1 if greyscale is True
        """
        pass



    @abstractmethod
    def to_plottable_measurement(self, x):
        """returns the measurement sample in a format that could be plotted for visualization

        Args:
            x (numpy array): one measurement sample given by _get_x

        Returns:
            numpy array: the formatted measurement sample ready to be plotted
        """
        pass


    @abstractmethod
    def to_plottable_output(self, y):
        """returns the output or ground_truth sample in a format that could be plotted for visualization

        Args:
            y (numpy array): one ground_truth sample given by _get_y, or a sample with the same shape and characteristics 
                            (like the output of the model)

        Returns:
            numpy array: the formatted output/ground_truth sample ready to be plotted
        """
        pass





def np_load(filename):
        return np.load(filename)


class WallerlabDaloader(DataLoader):

    def __init__(self, dataset_config, indexes, batch_size, greyscale=False, use_crop=True, seed=1):
        super().__init__(dataset_config, indexes, batch_size, greyscale, seed)
        self.crop = dataset_config['crop']['measurements'] if use_crop else None
        # self.raw_shape_measure = np.load(self.x_filenames[0]).shape
        # self.raw_shape_truth = np.load(self.y_filenames[0]).shape


    def _get_filenames(self):
        dir_x = os.path.join(self.data_conf['path'], self.data_conf['measure_folder'])
        x_filenames = sorted([name for name in os.listdir(dir_x) 
                             if name.endswith(self.data_conf['measure_format'])])

        dir_y =  os.path.join(self.data_conf['path'], self.data_conf['truth_folder'])
        y_filenames = sorted([name for name in os.listdir(dir_y)
                              if name.endswith(self.data_conf['truth_format'])])

        assert x_filenames == y_filenames, "the lensed filenames must be equal to the lensless filenames"

        x_filenames = np.asarray([os.path.join(dir_x, name) for name in x_filenames])
        y_filenames = np.asarray([os.path.join(dir_y, name) for name in y_filenames])
        return x_filenames, y_filenames

    
    
    def _map_x(self, data_measure):
        data_measure = data_measure.map(lambda item: tf.numpy_function(np_load, [item], tf.float32), 
                                        num_parallel_calls=tf.data.AUTOTUNE)
        if self.greyscale:
            data_measure = data_measure.map(lambda item : tf.py_function(tf_rgb2gray, [item], tf.float32), 
                                            num_parallel_calls=tf.data.AUTOTUNE)
        return data_measure#.map(lambda item: tf.transpose(item, perm=[2, 0, 1]), num_parallel_calls=tf.data.AUTOTUNE)

    
    def _map_y(self, data_truth):
        data_truth = self._map_x(data_truth,)
        if self.crop:
            data_truth = data_truth.map(lambda item: item[self.crop['low_h']: self.crop['high_h'],
                                                        self.crop['low_w']: self.crop['high_w'], :],
                                                        num_parallel_calls=tf.data.AUTOTUNE)
        return data_truth
    


    def to_plottable_measurement(self, x):
        return x / x.max()

    def to_plottable_output(self, y):
        return np.flipud(y) / y.max()
    



class PhlatnetDataLoader(DataLoader):
    def __init__(self, dataset_config, indexes, batch_size, greyscale=False, use_crop=True, seed=1):
        self.crop = dataset_config['crop']['measurements'] if use_crop else None
        self.use_cropped_dataset = dataset_config['use_cropped_dataset']
        self.downsample = dataset_config['downsample']
        super().__init__(dataset_config, indexes, batch_size, greyscale, seed)
        
        # self.raw_shape_measure = np.load(self.x_filenames[0]).shape
        # self.raw_shape_truth = np.load(self.y_filenames[0]).shape


    def _get_filenames(self):
        if not self.use_cropped_dataset:
            dir_x = os.path.join(self.data_conf['path'], self.data_conf['measure_folder'])
            x_filenames = sorted(glob.glob(dir_x + '/*/*'), key=lambda f: os.path.basename(f).replace('.png','').replace('.', ''))
            x_names = [os.path.basename(f).replace('.png','').replace('.', '') for f in x_filenames]
        else:
            dir_x = os.path.join(self.data_conf['path'], self.data_conf['measure_cropped_folder'])
            x_filenames = sorted(glob.glob(dir_x + '/*/*'), key=lambda f: os.path.basename(f).replace('.npy',''))
            x_names = [os.path.basename(f).replace('.npy','') for f in x_filenames]
        
        dir_y = os.path.join(self.data_conf['path'], self.data_conf['truth_folder'])
        y_filenames = sorted(glob.glob(dir_y + '/*/*'), key=lambda f: os.path.basename(f).replace('.JPEG',''))
        y_names = [os.path.basename(f).replace('.JPEG','') for f in y_filenames]

        print(len(set(x_names).symmetric_difference(set(y_names))))
        print(len(y_names), len(x_names))
        if not x_names == y_names:
            print('some of the samples do not match : list(groundtruths) != list(measurements), ',
                  'x length:', len(x_names), 'y length:', len(y_names), 
                  'diff length:', len(set(x_names).symmetric_difference(set(y_names))))
            y_filenames = [f for f in y_filenames if os.path.basename(f).replace('.JPEG','') in x_names]

            y_names = [os.path.basename(f).replace('.JPEG','') for f in y_filenames]
            print('final length:'  , len(y_names))
            assert x_names == y_names, 'some of the samples do not match : list(groundtruths) != list(measurements), '

        return np.asarray(x_filenames), np.asarray(y_filenames)

    
    
    def _map_x(self, data_measure):
         # -1 :return the loaded image as is (with alpha channel, otherwise it gets cropped) 
        # read as uint16, channel last
        if not self.use_cropped_dataset:
            raise NotImplementedError
            # data_measure = data_measure.map(lambda item: tf.numpy_function(extract_bayer_raw, [item], tf.float32))
            # data_measure = data_measure.map(lambda item: item[:, 
            #                                               self.crop['low_h']: self.crop['high_h'],
            #                                             self.crop['low_w']: self.crop['high_w']],
            #                                             num_parallel_calls=tf.data.AUTOTUNE)
        else:
            data_measure = data_measure.map(lambda item: tf.cast(tf.numpy_function(np_load, [item], tf.uint16) / MAX_UINT16_VAL, tf.float32), 
                                            num_parallel_calls=tf.data.AUTOTUNE)         

        data_measure = data_measure.map(lambda item: (item -0.5) * 2, 
                                        num_parallel_calls=tf.data.AUTOTUNE)
        # print([self.in_dim[0] , self.in_dim[1]])
        # data_measure = data_measure.map(lambda item: tf.image.resize(item, 
        #                                                              size=[self.in_dim[0] // self.downsample , self.in_dim[1] //self.downsample]
        #                                                                   ),
        #                                 num_parallel_calls=tf.data.AUTOTUNE)
        data_measure = data_measure.map(lambda item: item[::self.downsample, ::self.downsample, :],
                                        num_parallel_calls=tf.data.AUTOTUNE)
        return data_measure


    
    def _map_y(self, data_truth):
        def cv_imread(path):
            img = cv2.imread(path.numpy().decode("utf-8"))[:, :, ::-1]/ MAX_UINT8_VAL
            return cv2.resize(img, (self.data_conf['truth_width'], self.data_conf['truth_height'])).astype(np.float32)
        
        # def preprocess(path):
        #     path = path.numpy().decode("utf-8") # .numpy() retrieves data from eager tensor
        #     img = cv2.imread(path)[:, :, ::-1] / MAX_UINT8_VAL
        #     img = cv2.resize(img, (self.data_conf['truth_width'], self.data_conf['truth_height']))
        #     img = (img - 0.5) * 2
        #     img = np.transpose(img, (2, 0, 1))
        #     return img.astype(np.float32)
        
        data_truth = data_truth.map(lambda item: tf.py_function(cv_imread, [item], tf.float32),
                                   num_parallel_calls=tf.data.AUTOTUNE)
        
        # data_truth = data_truth.map(lambda item: (tf.transpose(item, perm=[2, 0, 1]) -0.5) * 2, 
        #                                 num_parallel_calls=tf.data.AUTOTUNE)
        data_truth = data_truth.map(lambda item: (item -0.5) * 2, 
                                        num_parallel_calls=tf.data.AUTOTUNE)
        return data_truth
    


    def to_plottable_measurement(self, x):
        return x / x.max()

    def to_plottable_output(self, y):
        return np.flipud(y) / y.max()





def get_tf_dataset(dataset_id, dataset_config, indexes, args):
    if dataset_id == 'wallerlab':
        return WallerlabDaloader(dataset_config=dataset_config,
                                  indexes=indexes,
                                  **args).get()
    # elif dataset_id in['flatnet', 'phlatnet'] :
    #     return PhlatnetDataGenerator(dataset_config=dataset_config, 
    #                                 indexes=indexes, 
    #                                  **args)
    elif dataset_id == 'phlatnet':
        return PhlatnetDataLoader(dataset_config=dataset_config,
                                  indexes=indexes,
                                  **args).get()
    else:
        raise NotImplementedError


