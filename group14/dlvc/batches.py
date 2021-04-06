import typing

from .dataset import Dataset
from .ops import Op

class Batch:
    '''
    A (mini)batch generated by the batch generator.
    '''

    def __init__(self):
        '''
        Ctor.
        '''

        self.data = None
        self.label = None
        self.idx = None

class BatchGenerator:
    '''
    Batch generator.
    Returned batches have the following properties:
      data: numpy array holding batch data of shape (s, SHAPE_OF_DATASET_SAMPLES).
      label: numpy array holding batch labels of shape (s, SHAPE_OF_DATASET_LABELS).
      idx: numpy array with shape (s,) encoding the indices of each sample in the original dataset.
    '''

    def __init__(self, dataset: Dataset, num: int, shuffle: bool, op: Op=None):
        '''
        Ctor.
        Dataset is the dataset to iterate over.
        num is the number of samples per batch. the number in the last batch might be smaller than that.
        shuffle controls whether the sample order should be preserved or not.
        op is an operation to apply to input samples.
        Raises TypeError on invalid argument types.
        Raises ValueError on invalid argument values, such as if num is > len(dataset).
        '''

        # TODO implement
        import numpy as np
        if num > len(dataset):
            raise ValueError('Invalid argument value: num > len(dataset).')
        if num < 1:
            raise ValueError('Invalid argument value: num < 1.')
        self.dataset = dataset
        self.num = num
        self.op = op

        self.idx = np.arange(len(self.dataset))
        if shuffle:
            np.random.shuffle(self.idx)

    def __len__(self) -> int:
        '''
        Returns the total number of batches the dataset is split into.
            This is identical to the total number of batches yielded every time the __iter__ method is called.
        '''

        # TODO implement
        return self.num

    def __iter__(self) -> typing.Iterable[Batch]:
        '''
        Iterate over the wrapped dataset, returning the data as batches.
        '''
        # TODO implement
        # The "yield" keyword makes this easier
        import numpy as np

        split_points = range(0, len(self.dataset), self.num)[1:]
        batches_idx = np.split(self.idx, split_points)

        for b in batches_idx:
            batch = Batch()
            batch.data = self.dataset.x[b]
            if self.op is not None:
                batch.data = np.asarray([self.op(i) for i in batch.data])
            batch.label = self.dataset.y[b]
            batch.idx = b
            yield batch


