import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.dummy import DummyClassifier

from ..model import Model


# noinspection DuplicatedCode
class KNN(Model):
    '''
    KNN classifier.
    Returns probabilistic class scores (all scores >= 0 and sum of scores = 1).
    '''

    def __init__(self, input_dim: int, num_classes: int, k: int):
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
        # wrong class
        if not isinstance(data, np.ndarray):
            raise TypeError("Invalid argument type for data. "
                            "Expected:{} ".format(np.ndarray),
                            "Provided:{}".format(type(data)))

        if not isinstance(labels, np.ndarray):
            raise TypeError("Invalid argument type for labels. "
                            "Expected:{} ".format(np.ndarray),
                            "Provided:{}".format(type(labels)))

        # wrong data types
        if not data.dtype == np.float32:
            raise TypeError("Invalid data type for data. "
                            "Expected:{} ".format(np.float32),
                            "Provided:{}".format(data.dtype))

        if not np.issubdtype(labels.dtype, np.integer):
            raise TypeError("Invalid data type for label. "
                            "Expected:{} ".format(np.integer),
                            "Provided:{}".format(labels.dtype))

        # wrong dimensions
        if not data.ndim == 2:
            raise ValueError("Invalid dimension for data. "
                             "Expected:{} ".format(2),
                             "Provided:{}".format(data.ndim))

        if not labels.ndim == 1:
            raise ValueError("Invalid dimension for labels. "
                             "Expected:{} ".format(1),
                             "Provided:{}".format(labels.ndim))
        # wrong shapes
        if not data.shape[0] == labels.shape[0]:
            raise ValueError("Dimension mismatch between data and labels. ",
                             "data shape:{} ".format(data.shape),
                             "labels shape:{}".format(labels.shape))

        if not data.shape[1] == self.input_dim:
            raise ValueError("Invalid 2nd dimension for data. "
                             "Expected:{} ".format(self.input_dim),
                             "Provided:{}".format(data.shape[1]))

        # wrong values
        if (np.min(labels) < 0) or (np.max(labels) >= self.num_classes):
            raise ValueError("Invalid values in labels. "
                             "Expected range:{}. ".format([0, self.num_classes-1]),
                             "Provided range:{}.".format([np.min(labels), np.max(labels)]))
        try:
            self.model.fit(data, labels)
            return 0.
        except:
            raise RuntimeError("Trololololo, RunTime Error.")

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
        # wrong class
        if not isinstance(data, np.ndarray):
            raise TypeError("Invalid argument type for data. "
                            "Expected:{} ".format(np.ndarray),
                            "Provided:{}".format(type(data)))

        # wrong data types
        if not data.dtype == np.float32:
            raise TypeError("Invalid data type for data. "
                            "Expected:{} ".format(np.float32),
                            "Provided:{}".format(data.dtype))

        # wrong dimensions
        if not data.ndim == 2:
            raise ValueError("Invalid dimension for data. "
                             "Expected:{} ".format(2),
                             "Provided:{}".format(data.ndim))

        # wrong shapes
        if not data.shape[1] == self.input_dim:
            raise ValueError("Invalid 2nd dimension for data. "
                             "Expected:{} ".format(self.input_dim),
                             "Provided:{}".format(data.shape[1]))

        try:
            return self.model.predict_proba(data)
        except:
            raise RuntimeError("Trololololo, RunTime Error.")


class SGD(Model):
    '''
    Simple classifier.
    Returns probabilistic class scores (all scores >= 0 and sum of scores = 1).
    '''

    def __init__(self, input_dim: int, num_classes: int, loss: str, alpha: float, max_iter: int):
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
        self.loss = loss
        self.model = SGDClassifier(loss=loss, alpha=alpha, max_iter=max_iter)

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
        # wrong class
        if not isinstance(data, np.ndarray):
            raise TypeError("Invalid argument type for data. "
                            "Expected:{} ".format(np.ndarray),
                            "Provided:{}".format(type(data)))

        if not isinstance(labels, np.ndarray):
            raise TypeError("Invalid argument type for labels. "
                            "Expected:{} ".format(np.ndarray),
                            "Provided:{}".format(type(labels)))

        # wrong data types
        if not data.dtype == np.float32:
            raise TypeError("Invalid data type for data. "
                            "Expected:{} ".format(np.float32),
                            "Provided:{}".format(data.dtype))

        if not np.issubdtype(labels.dtype, np.integer):
            raise TypeError("Invalid data type for label. "
                            "Expected:{} ".format(np.integer),
                            "Provided:{}".format(labels.dtype))

        # wrong dimensions
        if not data.ndim == 2:
            raise ValueError("Invalid dimension for data. "
                             "Expected:{} ".format(2),
                             "Provided:{}".format(data.ndim))

        if not labels.ndim == 1:
            raise ValueError("Invalid dimension for labels. "
                             "Expected:{} ".format(1),
                             "Provided:{}".format(labels.ndim))
        # wrong shapes
        if not data.shape[0] == labels.shape[0]:
            raise ValueError("Dimension mismatch between data and labels. ",
                             "data shape:{} ".format(data.shape),
                             "labels shape:{}".format(labels.shape))

        if not data.shape[1] == self.input_dim:
            raise ValueError("Invalid 2nd dimension for data. "
                             "Expected:{} ".format(self.input_dim),
                             "Provided:{}".format(data.shape[1]))

        # wrong values
        if (np.min(labels) < 0) or (np.max(labels) >= self.num_classes):
            raise ValueError("Invalid values in labels. "
                             "Expected range:{}. ".format([0, self.num_classes-1]),
                             "Provided range:{}.".format([np.min(labels), np.max(labels)]))
        try:
            self.model.fit(data, labels)
            return 0.
        except:
            raise RuntimeError("Trololololo, RunTime Error.")

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
        # wrong class
        if not isinstance(data, np.ndarray):
            raise TypeError("Invalid argument type for data. "
                            "Expected:{} ".format(np.ndarray),
                            "Provided:{}".format(type(data)))

        # wrong data types
        if not data.dtype == np.float32:
            raise TypeError("Invalid data type for data. "
                            "Expected:{} ".format(np.float32),
                            "Provided:{}".format(data.dtype))

        # wrong dimensions
        if not data.ndim == 2:
            raise ValueError("Invalid dimension for data. "
                             "Expected:{} ".format(2),
                             "Provided:{}".format(data.ndim))

        # wrong shapes
        if not data.shape[1] == self.input_dim:
            raise ValueError("Invalid 2nd dimension for data. "
                             "Expected:{} ".format(self.input_dim),
                             "Provided:{}".format(data.shape[1]))

        try:
            return self.model.predict_proba(data)
        except:
            raise RuntimeError("Trololololo, RunTime Error.")
