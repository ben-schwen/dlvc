from abc import ABCMeta, abstractmethod

import numpy as np

class PerformanceMeasure(metaclass=ABCMeta):
    '''
    A performance measure.
    '''

    @abstractmethod
    def reset(self):
        '''
        Resets internal state.
        '''

        pass

    @abstractmethod
    def update(self, prediction: np.ndarray, target: np.ndarray):
        '''
        Update the measure by comparing predicted data with ground-truth target data.
        Raises ValueError if the data shape or values are unsupported.
        '''

        pass

    @abstractmethod
    def __str__(self) -> str:
        '''
        Return a string representation of the performance.
        '''

        pass

    @abstractmethod
    def __lt__(self, other) -> bool:
        '''
        Return true if this performance measure is worse than another performance measure of the same type.
        Raises TypeError if the types of both measures differ.
        '''

        pass

    @abstractmethod
    def __gt__(self, other) -> bool:
        '''
        Return true if this performance measure is better than another performance measure of the same type.
        Raises TypeError if the types of both measures differ.
        '''

        pass


class Accuracy(PerformanceMeasure):
    '''
    Average classification accuracy.
    '''

    def __init__(self):
        '''
        Ctor.
        '''
        self.reset()

    def reset(self):
        '''
        Resets the internal state.
        '''
        # TODO implement
        self.correct = 0
        self.total = 0

    def update(self, prediction: np.ndarray, target: np.ndarray):
        '''
        Update the measure by comparing predicted data with ground-truth target data.
        prediction must have shape (s,c) with each row being a class-score vector.
            The predicted class label is the one with the highest probability.
        target must have shape (s,) and values between 0 and c-1 (true class labels).
        Raises ValueError if the data shape or values are unsupported.
        '''

        # TODO implement
        if not prediction.shape[0] == target.shape[0]:
            raise ValueError('Incomparable shapes of prediction and target.')

        pred_class = np.argmax(prediction, axis=1)
        self.correct = self.correct + np.sum(pred_class == target)
        self.total = self.total + len(target)

    def __str__(self):
        '''
        Return a string representation of the performance.
        '''

        # TODO implement
        # return something like "accuracy: 0.395"
        return 'accuracy: {}'.format(self.accuracy())


    def __lt__(self, other) -> bool:
        '''
        Return true if this accuracy is worse than another one.
        Raises TypeError if the types of both measures differ.
        '''

        # See https://docs.python.org/3/library/operator.html for how these
        # operators are used to compare instances of the Accuracy class
        # TODO implement
        if not isinstance(other, Accuracy):
            raise TypeError('Invalid type for other.')

        return self.accuracy() < other.accuracy()

    def __gt__(self, other) -> bool:
        '''
        Return true if this accuracy is better than another one.
        Raises TypeError if the types of both measures differ.
        '''

        # TODO implement
        if not isinstance(other, Accuracy):
            raise TypeError('Invalid type for other.')

        return self.accuracy() > other.accuracy()

    def accuracy(self) -> float:
        '''
        Compute and return the accuracy as a float between 0 and 1.
        Returns 0 if no data is available (after resets).
        '''

        # TODO implement
        # on this basis implementing the other methods is easy (one line)
        if self.total == 0:
            return 0
        else:
            return self.correct / self.total
