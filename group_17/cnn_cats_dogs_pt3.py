from dlvc.batches import BatchGenerator
from dlvc.dataset import Subset
from dlvc.datasets.pets import PetsDataset
import dlvc.ops as ops
import numpy as np
from dlvc.models.pytorch import CnnClassifier
import torch.nn as nn
import torch.nn.functional as F
from dlvc.test import Accuracy
import torch
import os
from torchvision import models

# 1. Load the training, validation, and test sets as individual PetsDatasets.

fp_ben = '/home/bschwendinger/github/cifar-10-python/cifar-10-batches-py/'
fp_fab = 'C:\\Users\\fabia\\OneDrive\\Dokumente\\Uni\\Deep Learning for Visual Computing\\Assignment1\\Datensatz\\cifar-10-python'
fp_server = '/caa/Student/dlvc/public/datasets/cifar-10/'

if os.path.isdir(fp_server):
    fp = fp_server
elif os.path.isdir(fp_ben):
    fp = fp_ben
elif os.path.isdir(fp_fab):
    fp = fp_fab

print("Lade Bilder")
train_ds = PetsDataset(fp, Subset.TRAINING)
valid_ds = PetsDataset(fp, Subset.VALIDATION)
test_ds = PetsDataset(fp, Subset.TEST)
print("Bilder geladen")

# 2. Create a BatchGenerator for each one. Traditional classifiers don't usually train in batches so you can set the
# minibatch size equal to the number of dataset samples to get a single large batch - unless you choose a classifier
# that does require multiple batches.
op = ops.chain([
    ops.add(-127.5),
    ops.mul(1 / 127.5),
    ops.hflip(),
    #ops.rotate90(),
    ops.rcrop(32, 4, pad_mode='constant'),
    ops.hwc2chw()
])

# set seed for reproducibility
np.random.seed(373)

batch_size = 64
train_b = BatchGenerator(train_ds, batch_size, True, op)
valid_b = BatchGenerator(valid_ds, 1024, True, op)
test_b = BatchGenerator(test_ds, batch_size, False, op)


# View images
def imshow(inp, title=None):
    import matplotlib.pyplot as plt
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # basic recipe
        out_size = 64

        self.conv1 = nn.Conv2d(3, out_size, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, stride=1, padding=1)

        in_size = out_size
        out_size *= 2

        self.conv3 = nn.Conv2d(in_size, out_size, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(out_size, out_size, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        size = int(8 * 8 * out_size)
        self.fc1 = nn.Linear(size, 2)

        self.dropout = nn.Dropout(0.2)

        self.out_size = out_size

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)

        x = x.view(-1, int(8 * 8 * self.out_size))
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        return x


net = Net()

if torch.cuda.is_available():
    net = net.cuda()

# print("== Training from scratch ===")
# clf = CnnClassifier(net, (0, 3, 32, 32), 2, lr=0.001, wd=0)
#
# for epoch in range(100):
#     print("epoch {}".format(epoch))
#
#     losses = []
#     for batch in iter(train_b):
#         losses.append(clf.train(batch.data, batch.label))
#     losses = np.array(losses)
#     print("\ttrain loss: {:.3f} +- {:.3f}".format(losses.mean(), losses.std()))
#
#     acc = Accuracy()
#     for batch in iter(valid_b):
#         acc.update(clf.predict(batch.data), batch.label)
#     print("\tval acc: accuracy: {:.3f}".format(acc.accuracy()))
#
print("Transfer learning")
net = models.resnet101(pretrained=True)
for name, param in net.named_parameters():
    if "bn" not in name:
        param.requires_grad = False
num_ftrs = net.fc.in_features
num_classes = 2

net.fc = nn.Sequential(nn.Linear(num_ftrs, 512),
                          nn.ReLU(),
                          nn.Dropout(),
                          nn.Linear(512, num_classes))

if torch.cuda.is_available():
    net = net.cuda()

clf = CnnClassifier(net, (0, 3, 32, 32), 2, lr=0.001, wd=0)
for epoch in range(100):
    print("epoch {}".format(epoch))

    losses = []
    for batch in iter(train_b):
        losses.append(clf.train(batch.data, batch.label))
    losses = np.array(losses)
    print("\ttrain loss: {:.3f} +- {:.3f}".format(losses.mean(), losses.std()))

    acc = Accuracy()
    for batch in iter(valid_b):
        acc.update(clf.predict(batch.data), batch.label)
    print("\tval acc: accuracy: {:.3f}".format(acc.accuracy()))
