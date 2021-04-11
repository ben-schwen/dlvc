import numpy as np
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier

from model import Model

class KNN(Model):
    '''
    Simple classifier.
    Returns probabilistic class scores (all scores >= 0 and sum of scores = 1).
    '''

    def __init__(self, input_dim: int, num_classes: int, k:int):
        '''
        Ctor.
        input_dim is the length of input vectors (> 0).
        num_classes is the number of classes (> 1).
        k is the number of nearest neighbors used for prediction (> 0).
        You are free to pass additional arguments necessary for the chosen classifier to this
            or other methods of this class (such as "k" for a Knn-classifier)
        '''
        # TODO implement
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.model = KNeighborsClassifier(n_neighbors=k)

    def input_shape(self) -> tuple:
        '''
        Returns the expected input shape as a tuple, which is (0, input_dim).
        '''

        # TODO implement
        return 0, self.input_dim

    def output_shape(self) -> tuple:
        '''
        Returns the shape of predictions for a single sample as a tuple, which is (num_classes,).
        '''

        # TODO implement
        return self.num_classes,

    def train(self, data: np.ndarray, labels: np.ndarray) -> float:
        '''
        Train the model on a batch of data.
        Data are the input data, with shape (m, input_dim) and type np.float32 (m is arbitrary).
        Labels has shape (m,) and integral values between 0 and num_classes - 1.
        Returns 0. as we are not yet training neural networks.
        Raises TypeError on invalid argument types.
        Raises ValueError on invalid argument values.
        Raises RuntimeError on other errors.
        '''

        # TODO implement
        self.model.fit(data, labels)

    def predict(self, data: np.ndarray) -> np.ndarray:
        '''
        Predict probabilistic class scores (all scores >= 0 and sum of scores = 1) from input data.
        Data are the input data, with a shape compatible with input_shape().
        The label array has shape (n, num_classes) with n being the number of input samples.
        Raises TypeError on invalid argument types.
        Raises ValueError on invalid argument values.
        Raises RuntimeError on other errors.
        '''

        # TODO implement
        return self.model.predict_proba(data)

class SGD(Model):
    '''
    Simple classifier.
    Returns probabilistic class scores (all scores >= 0 and sum of scores = 1).
    '''

    def __init__(self, input_dim: int, num_classes: int, alpha: float, max_iter: int):
        '''
        Ctor.
        input_dim is the length of input vectors (> 0).
        num_classes is the number of classes (> 1).
        k is the number of nearest neighbors used for prediction (> 0).
        You are free to pass additional arguments necessary for the chosen classifier to this
            or other methods of this class (such as "k" for a Knn-classifier)
        '''
        # TODO implement
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.model = SGDClassifier(loss="log", alpha=alpha, max_iter=max_iter)



    def input_shape(self) -> tuple:
        '''
        Returns the expected input shape as a tuple, which is (0, input_dim).
        '''

        # TODO implement
        return 0, self.input_dim

    def output_shape(self) -> tuple:
        '''
        Returns the shape of predictions for a single sample as a tuple, which is (num_classes,).
        '''

        # TODO implement
        return self.num_classes,

    def train(self, data: np.ndarray, labels: np.ndarray) -> float:
        '''
        Train the model on a batch of data.
        Data are the input data, with shape (m, input_dim) and type np.float32 (m is arbitrary).
        Labels has shape (m,) and integral values between 0 and num_classes - 1.
        Returns 0. as we are not yet training neural networks.
        Raises TypeError on invalid argument types.
        Raises ValueError on invalid argument values.
        Raises RuntimeError on other errors.
        '''

        # TODO implement
        self.model.fit(data, labels)

    def predict(self, data: np.ndarray) -> np.ndarray:
        '''
        Predict probabilistic class scores (all scores >= 0 and sum of scores = 1) from input data.
        Data are the input data, with a shape compatible with input_shape().
        The label array has shape (n, num_classes) with n being the number of input samples.
        Raises TypeError on invalid argument types.
        Raises ValueError on invalid argument values.
        Raises RuntimeError on other errors.
        '''

        # TODO implement
        return self.model.predict_proba(data)