import keras as K


class DuelingLayer(K.layers.Layer):

    def __init__(self, units: int, actions: int):
        self.v_layer1 = K.layers.Dense(units=units)
        self.v_activation = K.layers.LeakyReLU(alpha=0.3)
        self.v_layer2 = K.layers.Dense(units=1, activation='linear', input_dim=units)

        self.a_layer1 = K.layers.Dense(units=units)
        self.a_activation = K.layers.LeakyReLU(alpha=0.3)
        self.list_a = [K.layers.Dense(units=1, activation='linear', input_dim=units)
                       for _ in range(actions)]

        super(DuelingLayer, self).__init__()

    def __call__(self, input):
        v = self.v_layer1(input)
        v = self.v_activation(v)
        v = self.v_layer2(v)

        a = self.a_layer1(input)
        a = self.a_activation(a)
        a = [a_output(a) for a_output in self.list_a]

        average = K.layers.Average()(a)
        concat = K.layers.Concatenate()(a)
        add = K.layers.Add()([v, concat])
        aggregation_layer = K.layers.Subtract()([add, average])

        return aggregation_layer
