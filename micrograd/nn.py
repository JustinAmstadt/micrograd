import random
from micrograd.engine import Value

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron(Module):

    def __init__(self, nin, nonlin=True):
        # A list with the number of weights being all connections from the previous layer
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin

    # Forward pass
    # The length of incoming list x should be the same length as list self.w
    def __call__(self, x):
        # Taking the dot product of self.w and x then adding the bias
        # This looks like x1 * w1 + x2 * w2 ... xn * wn + b
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"

class Layer(Module):

    # nin: Each neuron in the layer expects 3 inputs
    # nout: 4 neurons in the layer
    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    # Each neutron gets the same inputs from the previous layer, but the weights and biases are what change things up
    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        # Only give us the value itself if the last value is of size 1
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):
    """
    nin: number of inputs into the whole MLP (i.e. number of input features)
    nouts: a list of integers like `[4, 5, 1]`, where each number is the number of neurons in each subsequent layer  
    → in this example: 3 layers: 4 neurons → 5 neurons → 1 neuron
    """
    def __init__(self, nin, nouts):
        # We need to add the number of neurons in the input layer (nin) to the list to get the full dimensions of the MLP
        # If nouts in [4, 5, 1] and nin is 3, the full dimensions are [3, 4, 5, 1]
        sz = [nin] + nouts

        # Creates each layer
        # In our example, it will make:
        # 3, 4
        # 4, 5
        # 5, 1

        # It also adds nonlinearity (e.g. ReLU) for each layer except the last one
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]

    def __call__(self, x):
        # Does the forward pass here
        # The output of one layer becomes the input of the next
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
