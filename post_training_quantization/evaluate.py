import numpy as np
import tensorflow as tf
from smh_utility_process_results import process_results
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.utils import to_categorical
from art.utils import load_dataset

class FakeQuantize(tf.keras.layers.Layer):
    
    def __init__(self, **kwargs):
        #print(kwargs)
        super().__init__(**kwargs)
    """
        self.input_low = kwargs['input_low']
        self.input_high = kwargs['input_high']
        self.output_low = kwargs['output_low']
        self.output_high = kwargs['output_high']
        self.levels = kwargs['levels']
    """ 
    def build(self, input_shape):
        return super().build(input_shape)

    def call(self, x):
        return x
        """
        if x <= min(self.input_low, self.input_high):
            output = self.output_low
        elif x > max(self.input_low, self.input_high):
            output = self.output_high
        else:
            output = round((x - self.input_low) / (self.input_high - self.input_low))
        return output
        """
    def compute_output_shape(self, input_shape):
        return input_shape


with tf.keras.utils.CustomObjectScope({'FakeQuantize': FakeQuantize}):
    model = tf.keras.models.load_model('test_cifar100_test_preset_performance_subset_size_300.h5')

_, (x_test, y_test) = cifar100.load_data()


input_tensor = tf.constant(x_test.astype('float32'))
predictions = np.argmax(model.predict(input_tensor), axis=1).reshape(y_test.shape)

print(predictions.shape)
print(y_test.shape)
process_results(predictions, y_test)
