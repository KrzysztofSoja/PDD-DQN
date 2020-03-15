import keras as K
import numpy as np

from gym.core import Env
from memory import AbstractMemory
from policy import AbstractPolicy
from sample import Sample
from typing import List, Tuple


class Agent:

    def __init__(self, environment: Env,
                 memory: AbstractMemory,
                 policy: AbstractPolicy,
                 model: K.Model,
                 gamma: float,
                 n_step: int = 1):
        self.environment = environment
        self.memory = memory
        self.policy = policy
        self.model = model
        self.current_model = None

        self.gamma = gamma
        self.n_step = n_step

        self.history = []

    def _explore_env(self, batch_size: int) -> Tuple[float, List[Sample]]:
        """ Return tuple of mean gain from all games and list of samples. """
        data = []
        gains = []
        state = self.environment.reset()
        previous_sample = None

        current_gain = 0
        for i in range(batch_size):
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
                gains.append(current_gain)
                current_gain = 0
                previous_sample = None
                state = self.environment.reset()
            else:
                state = next_state

        self.environment.close()
        return np.mean(gains)[0], data

    def _bellman_equation(self, batch: List[Sample]) -> np.ndarray:
        state = np.array([sample.state for sample in batch])
        q_values = self.current_model.predict(state)

        for idx in range(q_values.shape[0]):
            q_values[idx][batch[idx].action] = batch[idx].reward
            if not batch[idx].is_done():
                best_action_for_q_next = np.argmax(self.current_model.predict(batch[idx].next_state))
                q_next = self.model.predict(batch[idx].next_state)[0][best_action_for_q_next]
                q_values[idx][batch[idx].action] += self.gamma*q_next

        return q_values

    def learn(self, epochs: int,
              batch_size: int,
              change_model_delay: int,
              learning_rate: float):

        self.model.compile(loss='MSE', optimizer=K.optimizers.RMSprop(lr=learning_rate))
        self.current_model = K.models.clone_model(self.model)
        self.current_model.compile(loss='MSE', optimizer=K.optimizers.RMSprop(lr=learning_rate))

        history = []
        for epoch in range(epochs):
            print("Epoch: ", epoch)

            if epoch % change_model_delay == 0:
                self.model = K.models.clone_model(self.current_model)
                self.model.compile(loss='MSE', optimizer=K.optimizers.RMSprop(lr=learning_rate))

            eval_score, batch = self._explore_env(batch_size)
            self.memory.add(batch)
            batch = self.memory.sample(batch_size)

            q_values = self._bellman_equation(batch)
            state = np.array([sample.state for sample in batch])
            loss = self.current_model.fit(state, q_values, batch_size=batch_size)
            history.append({'loss': loss.history['loss'], 'eval_score': eval_score})
