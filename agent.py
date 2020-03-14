import keras as K
import numpy as np
import os

from gym.core import Env
from memory import AbstractMemory
from policy import AbstractPolicy
from sample import Sample
from typing import List


class Agent:

    def __init__(self, environment: Env, memory: AbstractMemory, policy: AbstractPolicy, model: K.Model):
        self.environment = environment
        self.memory = memory
        self.policy = policy
        self.model = model

    def _explore_env(self, batch_size: int) -> List[Sample]:
        data = []
        state = self.environment.reset()
        previous_sample = None

        for i in range(batch_size):
            q_values = self.model.predict(state)
            action = self.policy(q_values)

            next_state, reward, done, _ = self.environment.step(action)

            if previous_sample is None:
                previous_sample = Sample(state, action, next_state, reward)
            else:
                current_sample = Sample(state, action, next_state, reward)
                previous_sample.next_sample = current_sample
                data.append(previous_sample)
                previous_sample = current_sample

            if done:
                previous_sample = None
                state = self.environment.reset()
            else:
                state = next_state

        self.environment.close()
        return data

    def learn(self, epochs: int,
              batch_size: int,
              gamma: float,
              change_model_delay: int,
              learning_rate: int):

        self.model.compile(loss='MSE', optimizer=K.optimizers.RMSprop(lr=learning_rate))
        acc_model = K.models.clone_model(self.model)
        acc_model.compile(loss='MSE', optimizer=K.optimizers.RMSprop(lr=learning_rate))

        history = []
        for epoch in range(epochs):
            print("Epoch: ", epoch)

            if epoch % change_model_delay == 0:
                self.model = K.models.clone_model(acc_model.model)
                self.model.compile(loss='MSE', optimizer=K.optimizers.RMSprop(lr=learning_rate))

            self.memory.add(self._explore_env(batch_size))
            batch = self.memory.sample(batch_size)

            state = np.array([sample.state for sample in batch])
            # ToDo: Rozjebać te list compihension.
            q_values = np.array([[acc_model.predict(sample.state)[0][0],
                                  gamma * self.model.predict(sample.next_state)[0][
                                      np.argmax(acc_model.predict(sample.next_state))]]
                                 if sample.action == 1 else \
                                     [gamma * self.model.predict(sample.next_state)[0][
                                         np.argmax(acc_model.predict(sample.next_state))],
                                      acc_model.predict(sample.state)[0][1], ] \
                                 for sample in batch])

            Y = np.array([[0., 0.] if sample.next_sample is None else [1., 1.] for sample in batch])
            Y *= q_values
            Y += np.array(
                [[0, sample.reward] if sample.action == 1 else [sample.reward, 0] for sample in batch])

            loss = acc_model.fit(state, Y, batch_size=batch_size)
            #eval_score = self._eval_DQN(acc_model)
            #history.append({'loss': loss.history['loss'],
            #                'eval_score': eval_score})


