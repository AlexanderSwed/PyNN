import numpy as np
import matplotlib.pyplot as plt

import network as NN
import load_mnist as load


training_data, validation_data, test_data = load.load_data_wrapper()

net = NN.Network([784, 60, 10])
net.SGD(training_data, 30, 10, 0.5, test_data=test_data)
net.save('set.json')