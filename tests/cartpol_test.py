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

optimizer = K.optimizers.RMSprop(learning_rate=0.0005)

policy = Greedy()
memory = QueueMemory(maxlen=4_000)
agent = Agent(environment=environment,
              memory=memory,
              model=model,
              policy=policy,
              gamma=0.85,
              optimizer= optimizer,
              n_step=1)

agent.learn(epochs=100, batch_size=512, change_model_delay=15)
