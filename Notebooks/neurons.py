import numpy as np

def sigmoid(x):
    # função de ativação sigmoid: f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedforward(self, inputs):
        # Produto escalar de inputs e weights e adiciona o bias
        # E então usa a função de ativação sigmoid
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)

    def __str__(self):
        return f'Weights: {self.weights} | Bias: {self.bias}'

weights = np.array([0, 1]) # w1 = 0, w2 = 1
bias = 4                   # b = 4
n = Neuron(weights, bias)
print(n)

x = np.array([2, 3])                         
print(f'Neuron input = {x}')                 # x1 = 2, x2 = 3
print(f'Neuron output = {n.feedforward(x)}') # 0.9990889488055994