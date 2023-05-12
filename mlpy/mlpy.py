import numpy as np

class Perceptron:

    def __init__(self, input_size:int):
        self.w = []
        self.bias = np.random.randn()
        for i in range(input_size):
            self.w.append(np.random.randn())

    def fire(self, input):
        return np.inner(self.w, input) + self.bias

    def get_w(self):
        return self.w

    def get_bias(self):
        return self.bias

    def update_w(self, new_w, new_b):
        self.w = new_w
        self.b = new_b

class Layer:

    def __init__(self, input_size:int, size:int):
        self.size = size
        self.perceptrons = []
        for i in range(size):
            self.perceptrons.append(Perceptron(input_size))

class MLP:

    def __init__(self, input_size:int, n_layers:int, layers_sizes, activation='relu'):
        self.activation = activation
        self.layers = []
        self.input_size = input_size
        th = input_size

        for i in range(n_layers):
            self.layers.append(Layer(th, layers_sizes[i])):
            th = layer_sizes[i]

    def predict(self, input):
        next_layer_input = input
        
        for layer in self.layers:
            next_layer_input = [perceptron.fire() for perceptron in layer.perceptrons]
        
        return next_layer_input

    def get_w(self):
        w = []
        for layer in self.layers:
            for perceptron in layer.perceptrons:
                w.append()

def loss(y_hat, y):
    return 

def backpropagation(model, loss, dataset):
    preds = []
    for i in range(len(dataset)):
        X = dataset[i][:-1]
        y = datset[i][-1]
        y_hat = model.predict(X)

