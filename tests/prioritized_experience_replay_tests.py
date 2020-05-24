import unittest
import keras as K
import numpy as np

from memory import PrioritizedExperienceReplay
from sample import Sample
from model_wrapper import ModelWrapper
from typing import Tuple


class PrioritizedExperienceReplayTests(unittest.TestCase):

    @staticmethod
    def _create_model(input_shape: Tuple[int], output_size: int) -> K.models.Model:
        input = K.layers.Input(shape=input_shape)
        layer1 = K.layers.Dense(activation='relu', units=4)(input)
        layer2 = K.layers.Dense(units=output_size)(layer1)
        return K.models.Model(input, layer2)


    def test_can_fit_model(self):
        """ This test check ability of fitting model in PER to random vector. """
        state_shape = (4, )
        action_space = 2

        model = PrioritizedExperienceReplayTests._create_model(state_shape, action_space)
        PER = PrioritizedExperienceReplay(maxlen=1, model=model, key_scaling=10, gamma=1)
        model_wrapper = ModelWrapper(model=model, optimizer=K.optimizers.Adam(learning_rate=0.01))
        model_wrapper.compile()

        sample = Sample(action=np.random.randint(0, action_space),
                        state=np.random.rand(state_shape[0]),
                        reward=10,
                        next_state=None)
        PER.add(samples=[sample])

        history_of_loss = []
        fit_vector = np.zeros((action_space, ))
        fit_vector[sample.action] = sample.reward
        for _ in range(100):
            model_wrapper.fit(sample.state, fit_vector)
            history_of_loss.append(PER._loss_calculate(sample=sample))

        for idx, loss in enumerate(history_of_loss[:-1]):
            self.assertGreater(loss, history_of_loss[idx + 1])


