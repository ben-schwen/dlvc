import numpy as np
import torch
import torch.nn as nn
from torch import optim

from ..model import Model

class CnnClassifier(Model):
    '''
    Wrapper around a PyTorch CNN for classification.
    The network must expect inputs of shape NCHW with N being a variable batch size,
    C being the number of (image) channels, H being the (image) height, and W being the (image) width.
    The network must end with a linear layer with num_classes units (no softmax).
    The cross-entropy loss (torch.nn.CrossEntropyLoss) and SGD (torch.optim.SGD) are used for training.
    '''

    def __init__(self, net: nn.Module, input_shape: tuple, num_classes: int, lr: float, wd: float):
        '''
        Ctor.
        net is the cnn to wrap. see above comments for requirements.
        input_shape is the expected input shape, i.e. (0,C,H,W).
        num_classes is the number of classes (> 0).
        lr: learning rate to use for training (SGD with e.g. Nesterov momentum of 0.9).
        wd: weight decay to use for training.
        '''

        # TODO implement

        # Inside the train() and predict() functions you will need to know whether the network itself
        # runs on the CPU or on a GPU, and in the latter case transfer input/output tensors via cuda() and cpu().
        # To termine this, check the type of (one of the) parameters, which can be obtained via parameters() (there is an is_cuda flag).
        # You will want to initialize the optimizer and loss function here.
        # Note that PyTorch's cross-entropy loss includes normalization so no softmax is required
        self.net = net

        if not isinstance(input_shape, tuple):
            raise TypeError("Invalid argument type for input_shape. "
                            "Expected:{} ".format(type(input_shape)),
                            "Provided:{}".format(type(input_shape)))
        if not (len(input_shape) == 4):
            raise ValueError("Invalid length for input_shape. "
                             "Expected:{} ".format(4),
                             "Provided:{}".format(len(input_shape)))
        self.input_shape = input_shape

        if not isinstance(num_classes, int):
            raise TypeError("Invalid argument type for num_classes. "
                            "Expected:{} ".format(int),
                            "Provided:{}".format(type(num_classes)))
        if not (num_classes > 0):
            raise ValueError("Invalid value for num_classes. "
                             "Expected: num_classes > 0",
                             "Provided: num_classes = {}".format(num_classes))
        self.num_classes = num_classes

        self.lr = lr
        self.wd = wd

        self.cuda = next(net.parameters()).is_cuda

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr, weight_decay=self.wd, momentum=0.9, nesterov=True)

    def input_shape(self) -> tuple:
        '''
        Returns the expected input shape as a tuple.
        '''

        # TODO implement
        return self.input_shape

    def output_shape(self) -> tuple:
        '''
        Returns the shape of predictions for a single sample as a tuple, which is (num_classes,).
        '''

        # TODO implement
        return self.num_classes,

    def train(self, data: np.ndarray, labels: np.ndarray) -> float:
        '''
        Train the model on batch of data.
        Data has shape (m,C,H,W) and type np.float32 (m is arbitrary).
        Labels has shape (m,) and integral values between 0 and num_classes - 1.
        Returns the training loss.
        Raises TypeError on invalid argument types.
        Raises ValueError on invalid argument values.
        Raises RuntimeError on other errors.
        '''

        # TODO implement
        # Make sure to set the network to train() mode
        # See above comments on CPU/GPU

        if not isinstance(data, np.ndarray):
            raise TypeError('data is not of type np.ndarray')
        elif not isinstance(labels, np.ndarray):
            raise TypeError('labels is not of type np.ndarray')

        if data.shape[0] != labels.shape[0]:
            raise ValueError('Shapes of data and label differ')

        self.net.train()
        self.optimizer.zero_grad()

        # convert data to tensor
        data = (torch.from_numpy(data)).float()
        labels = (torch.from_numpy(labels)).long()

        # transfer to cuda
        if self.cuda:
            data = data.cuda()
            labels = labels.cuda()
        else:
            data = data.cpu()
            labels = labels.cpu()

        # forward pass
        output = self.net(data)
        loss = self.criterion(output, labels)

        # backward pass
        loss.backward()
        self.optimizer.step()

        self.net.eval()

        return loss.item()

    def predict(self, data: np.ndarray) -> np.ndarray:
        '''
        Predict softmax class scores from input data.
        Data has shape (m,C,H,W) and type np.float32 (m is arbitrary).
        The scores are an array with shape (n, output_shape()).
        Raises TypeError on invalid argument types.
        Raises ValueError on invalid argument values.
        Raises RuntimeError on other errors.
        '''

        # TODO implement

        # Pass the network's predictions through a nn.Softmax layer to obtain softmax class scores
        # Make sure to set the network to eval() mode
        # See above comments on CPU/GPU

        data = (torch.from_numpy(data)).float()

        # transfer to cuda
        if self.cuda:
            data = data.cuda()
        else:
            data = data.cpu()

        output = self.net(data)
        softmax = nn.Softmax(dim=1)

        # transfer back to cpu
        ans = softmax(output)

        return ans.cpu().detach().numpy()
