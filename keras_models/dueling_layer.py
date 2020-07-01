import keras as K


class DuelingLayer(K.layers.Layer):

    def __init__(self, units: int, actions: int,
                 kernel_regularizer: K.regularizers.Regularizer = None,
                 activity_regularizer: K.regularizers.Regularizer = None):
        self.v_layer1 = K.layers.Dense(units=units,
                                       kernel_regularizer=kernel_regularizer,
                                       activity_regularizer=activity_regularizer)
        self.v_activation = K.layers.LeakyReLU(alpha=0.3)
        self.v_layer2 = K.layers.Dense(units=1, activation='linear', input_dim=units,
                                       kernel_regularizer=kernel_regularizer,
                                       activity_regularizer=activity_regularizer)
        self.a_layer1 = K.layers.Dense(units=units,
                                       kernel_regularizer=kernel_regularizer,
                                       activity_regularizer=activity_regularizer)
        self.a_activation = K.layers.LeakyReLU(alpha=0.3)
        self.list_a = [K.layers.Dense(units=1, activation='linear', input_dim=units,
                                      kernel_regularizer=kernel_regularizer,
                                      activity_regularizer=activity_regularizer)
                       for _ in range(actions)]

        self.average = K.layers.Average()
        self.concat = K.layers.Concatenate()
        self.add = K.layers.Add()
        self.aggregation_layer = K.layers.Subtract()

        super(DuelingLayer, self).__init__()

    def __call__(self, input):
        v = self.v_layer1(input)
        v = self.v_activation(v)
        v = self.v_layer2(v)

        a = self.a_layer1(input)
        a = self.a_activation(a)
        a = [a_output(a) for a_output in self.list_a]

        average = self.average(a)
        a = self.concat(a)
        a = self.add([v, a])
        a = self.aggregation_layer([a, average])

        return a
