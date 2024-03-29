import keras as K
import gym

from agents.dqn import DQN
from memory import QueueMemory
from policy import EpsGreedy
from logger import Logger

logger = Logger(logdir_path='./test_result')
environment = gym.make('CartPole-v1')

input = K.layers.Input(shape=(4,))
batch_norm = K.layers.BatchNormalization()(input)
layer1 = K.layers.Dense(units=32, input_dim=4,
                        kernel_regularizer=K.regularizers.l2(0.01),
                        activity_regularizer=K.regularizers.l2(0.01))(batch_norm)
activation1 = K.layers.ReLU()(layer1)
layer2 = K.layers.Dense(units=32, input_dim=4,
                        kernel_regularizer=K.regularizers.l2(0.01),
                        activity_regularizer=K.regularizers.l2(0.01))(activation1)
activation2 = K.layers.ReLU()(layer2)
layer3 = K.layers.Dense(units=2, input_dim=4,
                        kernel_regularizer=K.regularizers.l2(0.01),
                        activity_regularizer=K.regularizers.l2(0.01))(activation2)
model = K.models.Model(input, layer3)

optimizer = K.optimizers.Adam(learning_rate=0.002)

policy = EpsGreedy(initial_eps=1.0, eps_min=0.05, eps_update=0.97)
memory = QueueMemory(maxlen=30_000)
agent = DQN(environment=environment,
            memory=memory,
            model=model,
            policy=policy,
            gamma=0.98,
            optimizer=optimizer,
            n_step=1,
            logger=logger)

agent.learn(epochs=1_000, batch_size_in_step=1024, change_model_delay=30,
            batch_size_in_exploration=5_000,
            min_n_game_in_exploration=10)
