import random
import json

import numpy as np

class Network(object):

    def __init__(self, sizes):
        """Список `sizes` содержит число нейронов сети. Для сети с тремя слоями
        первый элемент списка соответствует числу входных нейронов,
        второй - числу нейронов скрытого слоя, а третий - числу выходов.
        Очевидно, количество слоев задается размером списка. Веса и смещения задаются
        случайным образом, используя Гауссовское распределение на отрезке [-1; 1].
        Первый слой сети - входной слой, потому смещения инициализируются начиня со второго.
        Веса сети инициализируется для каждого промежутка между слоями."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def get_nn_output(self, a):
        """Возвращает выход сети для входа а."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Тренирует нейронную сеть, используя стохастический градиентный спуск.
        обучаемые данные описаны в модуле подготовки данных, epochs - количество
        эпох, которые нужно совершить до остановки обучения, mini_batch_size - размер
        мини-батча, eta - задает скорость обучения, learning rate.
        При передачи данных для теста выводит в консоль количество ошибок
        на тестовых данных."""
        points, points_1 = [], []
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    def feedforward(self, a):
        zs = []
        activations = [a]

        activation = a
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        return (zs, activations)


    def update_mini_batch(self, mini_batch, eta):
        batch_size = len(mini_batch)

        # метод преобразовывает каждый пример x/y из мини-батча в столбец
        # и заносит в массив входов/выходов
        X = np.asarray([x.ravel() for x, y in mini_batch]).transpose()
        Y = np.asarray([y.ravel() for x, y in mini_batch]).transpose()
        nabla_b, nabla_w = self.backprop(X, Y)
        self.weights = [w - (eta / batch_size) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / batch_size) * nb for b, nb in zip(self.biases, nabla_b)]


    def backprop(self, X, Y):
        nabla_b = [0 for i in self.biases]
        nabla_w = [0 for i in self.weights]
        zs, activations = self.feedforward(X)
        #delta = self.cost_derivative(activations[-1], Y) * sigmoid_prime(zs[-1])
        delta = activations[-1] - Y
        nabla_b[-1] = delta.sum(1).reshape([len(delta), 1])
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta.sum(1).reshape([len(delta), 1])
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())

        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Возвращает число правильных ответов на переданных данных"""
        test_results = [(np.argmax(self.feedforward(x)[1][-1]), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def save(self, filename):
        """Сохраняет параметры нейронной сети в файл filename"""
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases]}
        with open(filename, 'w') as f:
            json.dump(data, f)

####Дополнительные функции
def sigmoid(z):
    """Сигмоидная функция"""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Производная сигмоидной функции"""
    return sigmoid(z)*(1-sigmoid(z))

def load(filename):
    """Инициализирует сеть с загруженными параметрами
    """
    with open(filename, "r") as f:
        data = json.load(f)
    net = Network(data["sizes"])
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net