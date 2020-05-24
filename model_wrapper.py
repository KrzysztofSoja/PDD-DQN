import numpy as np
import keras as K


class ModelWrapper:
    """ Use to esthetically purpose. Make most method of class K.models.Model shorter. """

    def __init__(self, model: K.models.Model, optimizer: K.optimizers.Optimizer):
        self.model = model
        self.optimizer = optimizer

    def compile(self):
        self.model.compile(loss='MSE', optimizer=self.optimizer)

    def predict(self, x: np.array) -> float:
        if len(x.shape) == 1:
            x = np.array([x])
        return self.model.predict(x, batch_size=x.shape[0])

    def fit(self, x: np.array, y: np.array) -> float:
        if len(x.shape) == 1:
            x = np.array([x])
        if len(y.shape) == 1:
            y = np.array([y])

        return self.model.train_on_batch(x, y)

    def clone(self) -> object:
        clone_model = K.models.clone_model(self.model)
        return ModelWrapper(clone_model, self.optimizer)
