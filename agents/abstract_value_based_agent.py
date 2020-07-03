import keras as K

from gym.core import Env
from memory import AbstractMemory
from policy import AbstractPolicy
from model_wrapper import ModelWrapper
from sample import Sample

from typing import Tuple, List


class AbstractValueBasedAgent:

    def __init__(self, environment: Env,
                 memory: AbstractMemory,
                 policy: AbstractPolicy,
                 model: K.model,
                 optimizer: K.optimizers.Optimizer):
        self.environment = environment
        self.memory = memory
        self.policy = policy
        self.model = ModelWrapper(model, optimizer)

        self.history = []

    def _explore_env(self, batch_size: int, number_of_game: int = 10) -> Tuple[float, List[Sample]]:
        """ Return tuple of mean gain from all games and list of samples. """
        data = []
        gains = []
        state = self.environment.reset()
        previous_sample = None

        current_gain = 0
        n_sample = 0
        n_game = 0

        while n_game <= number_of_game or n_sample <= batch_size:
            q_values = self.model.predict(state)
            action = self.policy(q_values)

            next_state, reward, done, _ = self.environment.step(action)

            current_gain += reward

            if previous_sample is None:
                previous_sample = Sample(state, action, next_state, reward)
            else:
                current_sample = Sample(state, action, next_state, reward)
                previous_sample.next_sample = current_sample
                data.append(previous_sample)
                previous_sample = current_sample

            if done:
                data.append(current_sample)
                gains.append(current_gain)
                current_gain = 0
                previous_sample = None
                state = self.environment.reset()
                n_game += 1
            else:
                state = next_state
            n_sample += 1

    def learn(self, epochs: int,
              batch_size_in_step: int,
              min_n_game_in_exploration: int,
              batch_size_in_exploration: int,
              change_model_delay: int):
        raise NotImplementedError
