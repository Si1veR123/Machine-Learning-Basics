"""
VERY Basic custom neural network
No activation function
No loss function
"""

import random

class NeuralNetwork:
    def __init__(self, layers: int, count: list):
        self.num_layers = layers
        self.layers = self.make_neurons(count)

    def make_neurons(self, count):
        count.reverse()
        network = []
        prev_layer = []
        for x in range(self.num_layers):
            layer = []
            for _ in range(count[x]):
                layer.append(Neuron([prev_layer]))
            prev_layer = layer
            network.append(layer)
        network.reverse()
        return network

    def train(self, data, results, epochs):
        for _ in range(epochs):
            # data [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
            for d, r in zip(data, results):
                # check data is length of number of input neurons
                assert len(d) == len(self.layers[0])
                data_totals = []
                # 2nd layer will receive sum of inputs
                data_totals.append(sum(d))
                output = []
                # iterate over layers
                for count, layer in enumerate(self.layers[1:]):
                    layer_total = 0
                    for neuron in layer:
                        if layer != self.layers[-1]:
                            # calculate neuron's value by giving it total of previous layer
                            layer_total += neuron.calculate(data_totals[count])
                        else:
                            output.append(neuron.calculate(data_totals[count]))
                    data_totals.append(layer_total)
                if output[0] != r:
                    change = output[0] - r
                    upper = abs(int(change + change*0.1))
                    lower = abs(int(change - change*0.1))
                    for layer in self.layers:
                        for neuron in layer:
                            if output[0] > r:
                                neuron.weight -= random.randint(lower, upper)/50
                                #neuron.bias -= random.randint(lower, upper)//100
                            elif output[0] < r:
                                neuron.weight += random.randint(lower, upper)/50
                                #neuron.bias += random.randint(lower, upper)//100
                else:
                    change = 0

            print(f"""
            Data: {d}
            Pred: {output[0]}
            Result: {r}
            Change: {change}
            Upper: {upper/50}
            Lower: {lower/50}
            """)

    def predict(self, data):
        for d in data:
            # check data is length of number of input neurons
            assert len(d) == len(self.layers[0])
            data_totals = []
            # 2nd layer will receive sum of inputs
            data_totals.append(sum(d))
            output = []
            # iterate over layers
            for count, layer in enumerate(self.layers[1:]):
                layer_total = 0
                for neuron in layer:
                    if layer != self.layers[-1]:
                        # calculate neuron's value by giving it total of previous layer
                        layer_total += neuron.calculate(data_totals[count])
                    else:
                        output.append(neuron.calculate(data_totals[count]))
                data_totals.append(layer_total)
        return output

class Neuron:
    def __init__(self, connections):
        self.weight = 0.5
        self.bias = 0
        self.connections = connections

    def calculate(self, input_val):
        calculated = abs(self.bias+self.weight*input_val)
        return calculated


data = [[1, 2, 3], [4, 5, 6], [19, 20, 30], [8, 8, 8], [5, 5, 5], [1, 1, 1], [9, 8, 7], [4, 6, 7], [10, 10, 10]]

answers = []
for d in data:
    answers.append(sum(d))

print(answers)

nn = NeuralNetwork(3, [3, 2, 1])
nn.train(data, answers, 10)
print(round(nn.predict([[20, 33, 10]])[0], 0))