import unittest

import numpy as np

import dlvc.ops as ops
from dlvc.batches import BatchGenerator
from dlvc.dataset import Subset, Dataset
from dlvc.datasets.pets import PetsDataset
from dlvc.test import Accuracy

fp = 'C:\\Users\\bschwendinger\\Documents\\github\\dlvc\\cifar-10-batches-py'

class TestPets(unittest.TestCase):
    train = None
    valid = None
    test = None

    @classmethod
    def setUpClass(cls):
        cls.train = PetsDataset(fp, Subset.TRAINING)
        cls.valid = PetsDataset(fp, Subset.VALIDATION)
        cls.test = PetsDataset(fp, Subset.TEST)

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
        with self.assertRaises(IndexError):
            self.train[-1]
        with self.assertRaises(IndexError):
            self.train[9001]


class TestBatch(unittest.TestCase):
    train = None
    valid = None
    test = None

    @classmethod
    def setUpClass(cls):
        cls.train = PetsDataset(fp, Subset.TRAINING)
        cls.valid = PetsDataset(fp, Subset.VALIDATION)
        cls.test = PetsDataset(fp, Subset.TEST)

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

    def test_init_givenCase(self):
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

    def test_init_ValueError(self):
        with self.assertRaises(ValueError):
            BatchGenerator(self.train, 9000, False, None)
        with self.assertRaises(ValueError):
            BatchGenerator(self.train, 0, False, None)

    def test_init_TypeError(self):
        with self.assertRaises(TypeError):
            BatchGenerator("", 1, False, None)
        with self.assertRaises(TypeError):
            BatchGenerator(self.train, "string", False, None)
        with self.assertRaises(TypeError):
            BatchGenerator(self.train, 1, "", None)


class TestTest(unittest.TestCase):
    train = None

    @classmethod
    def setUpClass(cls):
        pets_ds = PetsDataset(fp, Subset.TRAINING)
        op = ops.chain([
            ops.vectorize(),
            ops.type_cast(np.float32),
            ops.add(-127.5),
            ops.mul(1 / 127.5),
        ])
        cls.train = next(iter(BatchGenerator(pets_ds, len(pets_ds), False, op)))

    @classmethod
    def tearDownClass(cls):
        cls.train = None

    def test_init(self):
        acc = Accuracy()
        self.assertEqual(acc.correct, 0)
        self.assertEqual(acc.total, 0)

    def test_accuracy(self):
        acc = Accuracy()
        self.assertEqual(acc.accuracy(), 0.0)

    def test_update_allCorrect(self):
        acc = Accuracy()
        preds = np.transpose(np.vstack([self.train.label == 0, self.train.label == 1]))
        acc.update(preds, self.train.label)
        self.assertEqual(acc.accuracy(), 1.0)

    def test_update_allWrong(self):
        acc = Accuracy()
        preds = np.transpose(np.vstack([self.train.label == 1, self.train.label == 0]))
        acc.update(preds, self.train.label)
        self.assertEqual(acc.accuracy(), 0.0)

    def test_update_valueError(self):
        with self.assertRaises(ValueError):
            acc = Accuracy()
            acc.update(self.train.label, self.train.label)

    def test_reset(self):
        acc = Accuracy()
        preds = np.transpose(np.vstack([self.train.label == 0, self.train.label == 1]))
        acc.update(preds, self.train.label)
        acc.reset()
        self.assertEqual(acc.accuracy(), 0.0)

    def test_lt(self):
        lower = Accuracy()
        higher = Accuracy()

        preds = np.transpose(np.vstack([self.train.label == 0, self.train.label == 1]))
        higher.update(preds, self.train.label)

        self.assertTrue(lower < higher)
        self.assertFalse(lower < lower)
        self.assertFalse(higher < higher)
        with self.assertRaises(TypeError):
            lower < True

    def test_gt(self):
        lower = Accuracy()
        higher = Accuracy()

        preds = np.transpose(np.vstack([self.train.label == 0, self.train.label == 1]))
        higher.update(preds, self.train.label)

        self.assertTrue(higher > lower)
        self.assertFalse(lower > lower)
        self.assertFalse(higher > higher)
        with self.assertRaises(TypeError):
            lower > True



if __name__ == '__main__':
    unittest.main()