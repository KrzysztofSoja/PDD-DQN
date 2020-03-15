import keras as K
import gym

from agent import Agent
from memory import QueueMemory
from policy import Greedy
from keras_models.dueling_layer import DuelingLayer


environment = gym.make('CartPole-v1')

input = K.layers.Input(shape=(4,))
layer = K.layers.Dense(units=32, input_dim=4)(input)
activation = K.layers.LeakyReLU(alpha=0.3)(layer)
dueling = DuelingLayer(actions=2, units=16)(activation)
model = K.models.Model(input, dueling)

policy = Greedy()
memory = QueueMemory(size=20_000)
agent = Agent(environment=environment,
              memory=memory,
              model=model,
              policy=policy,
              gamma=0.85,
              n_step=1)

agent.learn(epochs=200, batch_size=512, change_model_delay=120, learning_rate=0.0005)
