import numpy as np

from ..dataset import Sample, Subset, ClassificationDataset

class PetsDataset(ClassificationDataset):
    '''
    Dataset of cat and dog images from CIFAR-10 (class 0: cat, class 1: dog).
    '''

    def __init__(self, fdir: str, subset: Subset):
        '''
        Loads a subset of the dataset from a directory fdir that contains the Python version
        of the CIFAR-10, i.e. files "data_batch_1", "test_batch" and so on.
        Raises ValueError if fdir is not a directory or if a file inside it is missing.

        The subsets are defined as follows:
          - The training set contains all cat and dog images from "data_batch_1" to "data_batch_4", in this order.
          - The validation set contains all cat and dog images from "data_batch_5".
          - The test set contains all cat and dog images from "test_batch".

        Images are loaded in the order the appear in the data files
        and returned as uint8 numpy arrays with shape (32, 32, 3), in BGR channel order.
        '''

        # TODO implement
        # See the CIFAR-10 website on how to load the data files
        import os
        import pickle

        def unpickle(file):
            with open(file, 'rb') as fo:
                ans = pickle.load(fo, encoding='bytes')
            return ans

        fnames = ''

        if subset == Subset.TRAINING:
            fnames = ['data_batch_{}'.format(i) for i in range(1, 5)]
        elif subset == Subset.VALIDATION:
            fnames = ['data_batch_5']
        elif subset == Subset.TEST:
            fnames = ['test_batch']

        fps = [*map(lambda x: os.path.join(fdir, x), fnames)]

        if not all(map(os.path.exists, fps)):
            raise ValueError("Either fdir is not a directory or a file is missing.")

        cifar_dicts = map(unpickle, fps)

        x = []
        y = []
        for d in cifar_dicts:
            x.extend(d[b'data'])
            y.extend(d[b'labels'])

        def _convert_img(raw):
            img = raw.reshape(-1, 3, 32, 32)
            img = np.einsum('abcd->acdb', img)
            return img[..., ::-1]

        cats_dogs_index = np.isin(y, [3, 5])
        x = _convert_img(np.asarray(x)[cats_dogs_index])
        y = np.asarray(y)[cats_dogs_index]
        y[y == 3] = 0
        y[y == 5] = 1

        self.x = x
        self.y = y


    def __len__(self) -> int:
        '''
        Returns the number of samples in the dataset.
        '''

        # TODO implement
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> Sample:
        '''
        Returns the idx-th sample in the dataset.
        Raises IndexError if the index is out of bounds. Negative indices are not supported.
        '''

        # TODO implement
        if (idx >= self.__len__()) or (idx < 0):
            error_msg = "Index out of bounds. Valid input range:{}. Provided index:{}".format([0, self.__len__()-1], idx)
            raise IndexError(error_msg)
        return Sample(idx, self.x[idx], self.y[idx])


    def num_classes(self) -> int:
        '''
        Returns the number of classes.
        '''

        # TODO implement
        return np.unique(self.y).__len__()
