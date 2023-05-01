
from abc import ABC, abstractmethod
from utils import *
from tensorflow.data import Dataset
import tensorflow as tf



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
        data_measure = Dataset.from_tensor_slices(self.x_filenames)
        data_measure = self._map_x(data_measure)

        data_truth = data_measure
        # .map(lambda item: tf.numpy_function(np.load, [item], tf.float32))
        # load ground truth
        # data_truth = Dataset.from_tensor_slices(self.y_filenames).map(lambda item: tf.numpy_function(
        #   np.load, [item], tf.float32))
        # .map(lambda y: tf.py_function(self._get_y, [y], tf.float32))
        # zip together
        data = Dataset.zip((data_measure, data_truth))
        # shuffle
        data = data.shuffle(buffer_size=self.num_files, seed=self.seed)
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



class WallerlabDaloader(DataLoader):

    def __init__(self, dataset_config, indexes, batch_size, greyscale=False, use_crop=True, seed=1):
        super().__init__(dataset_config, indexes, batch_size, greyscale, seed)
        self.use_crop = use_crop
        self.raw_shape_measure = np.load(self.x_filenames[0]).shape
        self.raw_shape_truth = np.load(self.y_filenames[0]).shape


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
        data_measure = data_measure.map(lambda item: tf.numpy_function(np.load, [item], tf.float32))
        
        return data_measure
    
        if self.greyscale:
            x = rgb2gray(x)
        # to channel first
        return x.transpose((2, 0, 1))
    
    

    def _get_y(self, filename):

        raw_data = tf.io.read_file(filename)
        y = np.frombuffer(raw_data.numpy(), dtype=np.float32).reshape(-1, *self.raw_shape_truth)
        # y = self._get_x(filename.decode())

        if self.crop:
            y = y[:, self.crop['low_h']: self.crop['high_h'],
                  self.crop['low_w']: self.crop['high_w']]
        return y
    

    def to_plottable_measurement(self, x):
        return x.transpose(1, 2, 0) / x.max()

    def to_plottable_output(self, y):
        return np.flipud(y.transpose(1, 2, 0)) / y.max()
    




def get_tf_dataset(dataset_id, dataset_config, indexes, args):
    if dataset_id == 'wallerlab':
        return WallerlabDaloader(dataset_config=dataset_config,
                                  indexes=indexes,
                                  **args).get()
    # elif dataset_id in['flatnet', 'phlatnet'] :
    #     return PhlatnetDataGenerator(dataset_config=dataset_config, 
    #                                 indexes=indexes, 
    #                                  **args)
    else:
        raise NotImplementedError


