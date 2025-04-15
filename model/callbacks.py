# #############################################################################
# custom_callbacks.py
# =================
# Author :
# Jonathan REYMOND [jonathan.reymond7@gmail.com]
# #############################################################################

import os
import numpy as np
import tensorflow as tf

from keras import backend
from keras.utils import io_utils
from tensorflow.python.platform import tf_logging as logging


try:
    import requests
except ImportError:
    requests = None

from keras.callbacks import Callback, ReduceLROnPlateau, LearningRateScheduler
import tensorflow_model_optimization as tfmot





class ReduceLROnPlateauCustom(Callback):
    """Reduce learning rate when a metric has stopped improving.

    Models often benefit from reducing the learning rate by a factor
    of 2-10 once learning stagnates. This callback monitors a
    quantity and if no improvement is seen for a 'patience' number
    of epochs, the learning rate is reduced.

    Example:

    ```python
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=5, min_lr=0.001)
    model.fit(X_train, Y_train, callbacks=[reduce_lr])
    ```

    Args:
        monitor: quantity to be monitored.
        factor: factor by which the learning rate will be reduced.
          `new_lr = lr * factor`.
        patience: number of epochs with no improvement after which learning rate
          will be reduced.
        verbose: int. 0: quiet, 1: update messages.
        mode: one of `{'auto', 'min', 'max'}`. In `'min'` mode,
          the learning rate will be reduced when the
          quantity monitored has stopped decreasing; in `'max'` mode it will be
          reduced when the quantity monitored has stopped increasing; in
          `'auto'` mode, the direction is automatically inferred from the name
          of the monitored quantity.
        min_delta: threshold for measuring the new optimum, to only focus on
          significant changes.
        cooldown: number of epochs to wait before resuming normal operation
          after lr has been reduced.
        min_lr: lower bound on the learning rate.
    """

    def __init__(
        self,
        optimizer,
        monitor="val_loss",
        factor=0.1,
        patience=10,
        verbose=0,
        mode="auto",
        min_delta=1e-4,
        cooldown=0,
        min_lr=0,
        **kwargs,
    ):
        super().__init__()

        self.monitor = monitor
        if factor >= 1.0:
            raise ValueError(
                "ReduceLROnPlateau does not support "
                f"a factor >= 1.0. Got {factor}"
            )
        if "epsilon" in kwargs:
            min_delta = kwargs.pop("epsilon")
            logging.warning(
                "`epsilon` argument is deprecated and "
                "will be removed, use `min_delta` instead."
            )
        self.factor = factor
        self.min_lr = min_lr
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0  # Cooldown counter.
        self.wait = 0
        self.best = 0
        self.mode = mode
        self.monitor_op = None
        self.optimizer = optimizer
        self._reset()

    def _reset(self):
        """Resets wait counter and cooldown counter."""
        if self.mode not in ["auto", "min", "max"]:
            logging.warning(
                "Learning rate reduction mode %s is unknown, "
                "fallback to auto mode.",
                self.mode,
            )
            self.mode = "auto"
        if self.mode == "min" or (
            self.mode == "auto" and "acc" not in self.monitor
        ):
            self.monitor_op = lambda a, b: np.less(a, b - self.min_delta)
            self.best = np.Inf
        else:
            self.monitor_op = lambda a, b: np.greater(a, b + self.min_delta)
            self.best = -np.Inf
        self.cooldown_counter = 0
        self.wait = 0

    def on_train_begin(self, logs=None):
        self._reset()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs["lr"] = backend.get_value(self.optimizer.lr)
        current = logs.get(self.monitor)
        if current is None:
            logging.warning(
                "Learning rate reduction is conditioned on metric `%s` "
                "which is not available. Available metrics are: %s",
                self.monitor,
                ",".join(list(logs.keys())),
            )

        else:
            if self.in_cooldown():
                self.cooldown_counter -= 1
                self.wait = 0

            if self.monitor_op(current, self.best):
                self.best = current
                self.wait = 0
            elif not self.in_cooldown():
                self.wait += 1
                if self.wait >= self.patience:
                    old_lr = backend.get_value(self.optimizer.lr)
                    if old_lr > np.float32(self.min_lr):
                        new_lr = old_lr * self.factor
                        new_lr = max(new_lr, self.min_lr)
                        backend.set_value(self.optimizer.lr, new_lr)
                        if self.verbose > 0:
                            io_utils.print_msg(
                                f"\nEpoch {epoch +1}: "
                                "ReduceLROnPlateau reducing "
                                f"learning rate to {new_lr}."
                            )
                        self.cooldown_counter = self.cooldown
                        self.wait = 0

    def in_cooldown(self):
        return self.cooldown_counter > 0
    



class LearningRateSchedulerCustom(Callback):
    """Learning rate scheduler.

    At the beginning of every epoch, this callback gets the updated learning
    rate value from `schedule` function provided at `__init__`, with the current
    epoch and current learning rate, and applies the updated learning rate on
    the optimizer.

    Args:
      schedule: a function that takes an epoch index (integer, indexed from 0)
          and current learning rate (float) as inputs and returns a new
          learning rate as output (float).
      verbose: int. 0: quiet, 1: update messages.

    Example:

    >>> # This function keeps the initial learning rate for the first ten epochs
    >>> # and decreases it exponentially after that.
    >>> def scheduler(epoch, lr):
    ...   if epoch < 10:
    ...     return lr
    ...   else:
    ...     return lr * tf.math.exp(-0.1)
    >>>
    >>> model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
    >>> model.compile(tf.keras.optimizers.SGD(), loss='mse')
    >>> round(model.optimizer.lr.numpy(), 5)
    0.01

    >>> callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
    >>> history = model.fit(np.arange(100).reshape(5, 20), np.zeros(5),
    ...                     epochs=15, callbacks=[callback], verbose=0)
    >>> round(model.optimizer.lr.numpy(), 5)
    0.00607

    """

    def __init__(self, optimizer, schedule, verbose=0):
        super().__init__()
        self.optimizer
        self.schedule = schedule
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')
        try:  # new API
            lr = float(backend.get_value(self.optimizer.lr))
            lr = self.schedule(epoch, lr)
        except TypeError:  # Support for old API for backward compatibility
            lr = self.schedule(epoch)
        if not isinstance(lr, (tf.Tensor, float, np.float32, np.float64)):
            raise ValueError(
                'The output of the "schedule" function '
                f"should be float. Got: {lr}"
            )
        if isinstance(lr, tf.Tensor) and not lr.dtype.is_floating:
            raise ValueError(
                f"The dtype of `lr` Tensor should be float. Got: {lr.dtype}"
            )
        backend.set_value(self.model.optimizer.lr, backend.get_value(lr))
        if self.verbose > 0:
            io_utils.print_msg(
                f"\nEpoch {epoch + 1}: LearningRateScheduler setting learning "
                f"rate to {lr}."
            )

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs["lr"] = backend.get_value(self.optimizer.lr)



class CopyLearningRate(Callback):
    def __init__(self, optimizer_listener, optimizer_target):
        """copy the learning rate from one optimizer to another

        Args:
            optimizer_listener (keras.Optimizer): optimizer to listen to
            optimizer_target (keras.Optimizer): optimizer to copy the learning rate to
        """
        super().__init__()
        self.optimizer_listener = optimizer_listener
        self.optimizer_target = optimizer_target

    def on_epoch_begin(self, epoch, logs=None):
        backend.set_value(self.optimizer_listener.lr, float(backend.get_value(self.optimizer_target.lr)))
        
        
class ChangeLossWeights(Callback):
    """Change the weights of the losses during training
    
    Args:
        weights_factors (list): list of tuples (weight, additive_factor) to add to the weight
    """
    def __init__(self, weights_factors):
        self.weights_factors = weights_factors

    def on_epoch_end(self, epoch, logs=None):
        for weight, additive_factor in self.weights_factors:
            if weight + additive_factor < 0 :
                print('\nno weight update, current minus value :', weight)
            else:
                backend.set_value(weight, weight + additive_factor)





def get_callbacks(model, store_folder, checkpoint_path, dynamic_weights, config):
    """Get callbacks for training
    
    Args:
        model (keras.Model): model to train
        store_folder (str): folder to store the logs
        checkpoint_path (str): path to store the checkpoints
        dynamic_weights (list): list of tuples (weight, additive_factor) to add to the weight
        config (dict): configuration dictionnary"""

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
                                            filepath=checkpoint_path,
                                            save_weights_only=True,
                                            monitor='val_total',
                                            mode='min',
                                            save_best_only=True,
                                            save_freq="epoch",
                                            verbose=1)
    
    callbacks = [model_checkpoint]

    if config['weight_pruning']:
        callbacks.append(tfmot.sparsity.keras.UpdatePruningStep())
        callbacks.append(tfmot.sparsity.keras.PruningSummaries(log_dir=os.path.join(store_folder, 'pruning_logs')))

    if dynamic_weights:
        print('Using dynamic weights')
        callbacks.append(ChangeLossWeights(dynamic_weights))

    if config['lr_reducer']['type']:
        reducer_args = config['lr_reducer']
        if reducer_args['type'] == "reduce_lr_on_plateau":
            callbacks.append(ReduceLROnPlateau(**reducer_args['reduce_lr_on_plateau']))
        elif reducer_args['type'] == "learning_rate_scheduler":

            def lr_scheduler(epoch, lr, epochs_interval, factor, min_lr):
                if (epoch +1) % epochs_interval == 0 and epoch > 0:
                    return max(lr * factor, min_lr)
                else:
                    return lr
                
            scheduler = lambda epoch, lr: lr_scheduler(epoch=epoch, lr=lr, **reducer_args['learning_rate_scheduler'])
            callbacks.append(LearningRateScheduler(scheduler, verbose=1))
        else:
            raise ValueError(reducer_args['type'] + " type is not supported, must be either 'reduce_lr_on_plateau' or 'learning_rate_scheduler'")
        
    if config['use_discriminator']:
        assert not (config['discriminator']['optimizer']['use_lr_reducer'] and config['discriminator']['optimizer']['copy_gen_lr']), 'Cannot use both lr reducer and copy gen lr' 
        if config['discriminator']['optimizer']['use_lr_reducer']:
            print('Using discriminator lr reducer')
            reducer_args = config['discriminator']['optimizer']['lr_reducer']
            if reducer_args['type'] == "reduce_lr_on_plateau":
                callbacks.append(ReduceLROnPlateauCustom(model.d_optimizer, **reducer_args['reduce_lr_on_plateau']))
            elif reducer_args['type'] == "learning_rate_scheduler":
                callbacks.append(LearningRateSchedulerCustom(model.d_optimizer, **reducer_args['learning_rate_scheduler']))
            else:
                raise ValueError(reducer_args['type'] + " type is not supported, must be either 'reduce_lr_on_plateau' or 'learning_rate_scheduler'")
        elif config['discriminator']['optimizer']['copy_gen_lr']:
            print('Using discriminator lr copy')
            callbacks.append(CopyLearningRate(model.d_optimizer, model.optimizer))
            


    if config['use_tensorboard']:
        tb_path = os.path.join(store_folder, 'tensorboard_logs')
        if not os.path.isdir(tb_path):
            os.makedirs(tb_path)
        tb_callback = tf.keras.callbacks.TensorBoard(tb_path, 
                                                     histogram_freq = 1, 
                                                     profile_batch=config['tensorboard_profile_batch'],
                                                     update_freq='epoch')
        callbacks.append(tb_callback)
    
    return callbacks
