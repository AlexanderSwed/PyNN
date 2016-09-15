#### Libraries
# Standard library
import random
import json
import matplotlib.pyplot as plt

# Third-party libraries
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
        #self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        #self.weights = [np.random.randn(y, x)
        #                for x, y in zip(sizes[:-1], sizes[1:])]
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, a):
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
                self.update_mini_batch_m(mini_batch, eta)
            if test_data:
                r = self.evaluate(test_data)
                #tr = self.evaluate(training_data, True)
                points.append(r)
                #points_1.append(tr)
                print("Epoch {0}: {1} / {2}".format(
                    j, r, n_test))
            else:
                print("Epoch {0} complete".format(j))
        plt.plot(points, 'g')
        #plt.plot(points_1, 'g')
        plt.show()

    def update_mini_batch(self, mini_batch, eta):
        """Обновляет веса и смещения сети, применяя градиентный спуск, используя
        алгоритм обратного распространения для одного мини-батча."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def feedforward_m(self, a):
        zs = []
        activations = [a]

        activation = a
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        return (zs, activations)


    def update_mini_batch_m(self, mini_batch, eta):
        batch_size = len(mini_batch)

        # метод преобразовывает каждый пример x из мини-батча в столбец
        # и заносит в массив входов
        X = np.asarray([_x.ravel() for _x, _y in mini_batch]).transpose()
        # метод преобразовывает каждый выход y из мини-батча в столбец
        # и заносит в массив выходов
        Y = np.asarray([_y.ravel() for _x, _y in mini_batch]).transpose()

        nabla_b, nabla_w = self.backprop_m(X, Y)
        self.weights = [w - (eta / batch_size) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / batch_size) * nb for b, nb in zip(self.biases, nabla_b)]


    def backprop_m(self, X, Y):
        nabla_b = [0 for i in self.biases]
        nabla_w = [0 for i in self.weights]
        zs, activations = self.feedforward_m(X)
        #delta = self.cost_derivative(activations[-1], Y) * sigmoid_prime(zs[-1])
        delta = self.cost_derivative(activations[-1], Y)
        nabla_b[-1] = delta.sum(1).reshape([len(delta), 1])
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta.sum(1).reshape([len(delta), 1])
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())

        return (nabla_b, nabla_w)


    def backprop(self, x, y):
        """Вовзращает кортеж `(nabla_b, nabla_w)` отображающих градиент
         целевой функции C для примера x. nabla_b и nabla_w - список
         массивов ndarray, слой за слоем."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # прямое распространение
        activation = x
        activations = [x] # список всех активаций, слой за слоем
        zs = [] # список векторов z, слой за слоем
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # обратное распространение
        #delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1]) #старая формула
        delta = self.cost_derivative(activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for layer in range(2, self.num_layers):
            z = zs[-layer]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-layer+1].transpose(), delta) * sp
            nabla_b[-layer] = delta
            nabla_w[-layer] = np.dot(delta, activations[-layer-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data, train = False):
        """Возвращает число правильных ответов на переданных данных"""
        #if train:
        #    test_results = [(np.argmax(self.feedforward(x)), np.argmax(y))
        #                for (x, y) in test_data]
        #else:
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, a, y):
        """Возвращает производную функции стоимости"""
        return (a-y)

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