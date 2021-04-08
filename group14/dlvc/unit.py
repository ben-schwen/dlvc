import unittest
import numpy as np

from dlvc.datasets.pets import PetsDataset
from dlvc.dataset import Subset
from dlvc.batches import Batch, BatchGenerator
import dlvc.ops as ops

class TestPets(unittest.TestCase):
    train = None
    valid = None
    test = None

    @classmethod
    def setUpClass(cls):
        cls.train = PetsDataset('C:\\Users\\bschwendinger\\Documents\\github\\dlvc\\cifar-10-batches-py', Subset.TRAINING)
        cls.valid = PetsDataset('C:\\Users\\bschwendinger\\Documents\\github\\dlvc\\cifar-10-batches-py', Subset.VALIDATION)
        cls.test = PetsDataset('C:\\Users\\bschwendinger\\Documents\\github\\dlvc\\cifar-10-batches-py', Subset.TEST)

    @classmethod
    def tearDownClass(cls):
        cls.train = None
        cls.valid = None
        cls.test = None

    def test_len(self):
        self.assertEqual(self.train.__len__(), 7959)
        self.assertEqual(self.valid.__len__(), 2041)
        self.assertEqual(self.test.__len__(), 2000)

    def test_numClasses(self):
        self.assertEqual(self.train.num_classes(), 2)
        self.assertEqual(self.valid.num_classes(), 2)
        self.assertEqual(self.test.num_classes(), 2)

    def test_getItem(self):
        self.assertEqual(self.train[0].shape, (32, 32, 3))
        self.assertEqual(self.valid[0].shape, (32, 32, 3))
        self.assertEqual(self.test[0].shape, (32, 32, 3))

    def test_getItem_Error(self):
        with self.assertRaises(IndexError) as ctx:
            self.train[-1]
        self.assertEqual('Index out of bounds.', str(ctx.exception))
        with self.assertRaises(IndexError) as ctx:
            self.train[9001]
        self.assertEqual('Index out of bounds.', str(ctx.exception))


class TestBatch(unittest.TestCase):
    train = None
    valid = None
    test = None

    @classmethod
    def setUpClass(cls):
        cls.train = PetsDataset('C:\\Users\\bschwendinger\\Documents\\github\\dlvc\\cifar-10-batches-py', Subset.TRAINING)
        cls.valid = PetsDataset('C:\\Users\\bschwendinger\\Documents\\github\\dlvc\\cifar-10-batches-py', Subset.VALIDATION)
        cls.test = PetsDataset('C:\\Users\\bschwendinger\\Documents\\github\\dlvc\\cifar-10-batches-py', Subset.TEST)

    @classmethod
    def tearDownClass(cls):
        cls.train = None
        cls.valid = None
        cls.test = None

    def test_init(self):
        b = BatchGenerator(self.train, 16, False, None)
        my_iter = iter(b)
        batch = next(my_iter)
        self.assertTrue(np.allclose(batch.label, np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0])))

    def test_givenInit(self):
        op = ops.chain([
            ops.vectorize(),
            ops.type_cast(np.float32),
            ops.add(-127.5),
            ops.mul(1 / 127.5),
        ])
        b = BatchGenerator(self.train, 500, False, op)
        my_iter = iter(b)
        batch = next(my_iter)
        given = np.array([-0.09019608, -0.01960784, -0.01960784, -0.28627452, -0.20784315])
        self.assertTrue(np.allclose(batch.data[0][0:5], given))


if __name__ == '__main__':
    unittest.main()