import pickle
import gzip

import numpy as np

def load_data():
    """Возвращает набор данных MNIST как кортеж состоящий из обучающих,
    проверочных и тестовых данных.
    Обучающие данные - 50,000 изображений и соответствующих подписей к ним.
    Проверочные данные и тестовые содержат по 10,000 изображений и подписей."""
    with gzip.open('data/mnist.pkl.gz', 'rb') as f:
        training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    """Возвращает кортеж, содержащий обучающие, проверочные и тестовые данные.
    Данная функция предоставляет более удобный формат для работы с нейронными сетями.
    Загруженные с помощью load_data() данные представляются в виде списка, содержащего
    кортежи размерности 2. На первом месте стоит само изображение в виде
    массива NumPy ndarray размерности 28х28=724 строк, 1 столбец. Под индексом 1 в кортеже
    содержится векторное представление ответа, предоставляемое функцией vectorized_result,
    описание которой находится в документации к самой функции."""
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    """Возвращает вектор размерности (10,1), где все элементы - 0,
    кроме j-го элемента, который равен 1 и соответсвует цифре из подписи
    к изображению. Данная функция нужна для преобразования цифры (0..9)
    в удобный для выходного слоя нашей нейронной сети векторный формат."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e