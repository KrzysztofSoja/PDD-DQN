import keras as K
import gym

from agent import Agent
from memory import PrioritizedExperienceReplay
from policy import Greedy
from keras_models.dueling_layer import DuelingLayer


environment = gym.make('CartPole-v1')

input = K.layers.Input(shape=(4,))
layer = K.layers.Dense(units=32, input_dim=4)(input)
activation = K.layers.ReLU()(layer)
dueling = DuelingLayer(actions=2, units=32)(activation)
model = K.models.Model(input, dueling)

optimizer = K.optimizers.Adam(learning_rate=0.005)

policy = Greedy()
memory = PrioritizedExperienceReplay(maxlen=4_000, model=model, key_scaling=10, gamma=0.85)
agent = Agent(environment=environment,
              memory=memory,
              model=model,
              policy=policy,
              gamma=0.85,
              optimizer=optimizer,
              n_step=1)

agent.learn(epochs=2_000, batch_size_in_step=64, change_model_delay=30,
            batch_size_in_exploration=512,
            min_n_game_in_exploration=10)
