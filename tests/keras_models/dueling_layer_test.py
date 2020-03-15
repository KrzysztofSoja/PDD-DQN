import unittest
import numpy as np
import keras as K
import tensorflow as tf

from keras_models.dueling_layer import DuelingLayer


class DuelingLayerTest(unittest.TestCase):

    def test_fitting(self):
        """ Testing ability to fitting module to embedding random vector. """
        input_layer = K.layers.Input(shape=(4,))
        layer = DuelingLayer(actions=2, units=4)(input_layer)
        test_model = K.models.Model(input_layer, layer)

        input = np.random.rand(3, 4)
        target = np.random.rand(3, 2)

        test_model.compile(loss='MSE', optimizer=K.optimizers.SGD(learning_rate=0.001))
        test_model.fit(input, target, batch_size=3, epochs=1000, verbose=0)
        output = test_model.predict(input, batch_size=3)

        self.assertTrue(np.mean(target - output)**2 < 0.01)


if __name__ == '__main__':
    unittest.main()