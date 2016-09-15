import network as NN
import numpy as np
import json
import matplotlib.pyplot as plt

with open('page/my_outfile.json', "r") as f:
    data = json.load(f)

x = np.asarray(data).reshape((784, 1))
plt.gray()
plt.matshow(np.reshape(x, (28, 28)))
plt.show()

net = NN.load('page/set.json')
print(np.argmax(net.get_nn_output(x)))