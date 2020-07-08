import keras as K
import numpy as np

from gym.core import Env
from memory import AbstractMemory, PrioritizedExperienceReplay
from policy import AbstractPolicy
from agents.abstract_agent import AbstractAgent
from logger import Logger
from sample import Sample
from model_wrapper import ModelWrapper
from typing import List
from tqdm import tqdm


class DQN(AbstractAgent):

    def __init__(self, environment: Env,
                 memory: AbstractMemory,
                 policy: AbstractPolicy,
                 model: K.Model,
                 logger: Logger,
                 gamma: float,
                 optimizer: K.optimizers.Optimizer,
                 n_step: int = 1):

        super(DQN, self).__init__(environment=environment, memory=memory, policy=policy,
                                  model=model, optimizer=optimizer, logger=logger)

        self.model = ModelWrapper(model, optimizer)
        self.current_model = None

        self.gamma = gamma
        self.n_step = n_step

    def _bellman_equation(self, batch: List[Sample]) -> np.ndarray:
        state = np.array([sample.state for sample in batch])
        q_values = self.current_model.predict(state)

        for idx in range(q_values.shape[0]):
            q_values[idx][batch[idx].action] = batch[idx].reward
            if not batch[idx].is_done():
                best_action_for_q_next = np.argmax(self.model.predict(batch[idx].next_state))
                q_next = self.model.predict(batch[idx].next_state)[0][best_action_for_q_next]
                q_values[idx][batch[idx].action] += self.gamma*q_next

        return q_values

    def learn(self, epochs: int,
              batch_size_in_step: int,
              min_n_game_in_exploration: int,
              batch_size_in_exploration: int,
              change_model_delay: int):

        self.model.compile()
        self.current_model = self.model.clone()
        self.current_model.compile()

        eval_score, starting_experience = self._explore_env(self.memory.maxlen)
        self.memory.add(starting_experience)

        for epoch in tqdm(range(epochs), desc='Learning in progress: '):

            if epoch % change_model_delay == 0:
                self.model = self.current_model.clone()
                self.model.compile()
                if type(self.memory) == PrioritizedExperienceReplay:
                    self.memory.update_model(self.model)
                eval_score, batch = self._explore_env(batch_size_in_exploration,
                                                      min_n_game_in_exploration)
                self.memory.add(batch)
            batch = self.memory.sample(batch_size_in_step)

            q_values = self._bellman_equation(batch)
            state = np.array([sample.state for sample in batch])
            loss = self.current_model.fit(state, q_values)
            self.policy.update()
            self.logger.add_event({'loss_value': loss, 'mean_gain': eval_score, 'epoch': epoch})


