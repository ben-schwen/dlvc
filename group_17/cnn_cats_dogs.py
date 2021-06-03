import datetime

from dlvc.batches import BatchGenerator
from dlvc.dataset import Subset
from dlvc.datasets.pets import PetsDataset
import dlvc.ops as ops
import numpy as np
from dlvc.models.pytorch import CnnClassifier
import torch.nn as nn
from dlvc.test import Accuracy
import torch
import os
import argparse
import json
from datetime import datetime

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

# set seed for reproducibility
torch.manual_seed(373)
np.random.seed(373)

if __name__ == '__main__':
    # Parse args
    parser = argparse.ArgumentParser(description='Lorem ipsum')
    parser.add_argument('--epoch', type=int, default=50, help='Number of epochs')
    parser.add_argument('--channel', action='store_true', help='Use per channel standardization')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--fc_size', type=int, default=256, help='Size of last linear layer')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    args = parser.parse_args()

    print("Lade Bilder")
    train_ds = PetsDataset(fp, Subset.TRAINING)
    valid_ds = PetsDataset(fp, Subset.VALIDATION)
    test_ds = PetsDataset(fp, Subset.TEST)
    print("Bilder geladen")

    # calculate per channel mean/sd
    # already works well since we didnt fuck with broadcasting
    full_train = next(iter(BatchGenerator(train_ds, len(train_ds), False)))
    mean = np.mean(full_train.data)
    std = np.std(full_train.data)
    mean_channel = np.mean(full_train.data, axis=(0, 1, 2))
    std_channel = np.std(full_train.data, axis=(0, 1, 2))

    # 2. Create a BatchGenerator for each one. Traditional classifiers don't usually train in batches so you can set the
    # minibatch size equal to the number of dataset samples to get a single large batch - unless you choose a classifier
    # that does require multiple batches.
    if (args.channel):
        print("Per channel normalization")
        op = ops.chain([
            ops.add(-mean_channel),
            ops.mul(1 / std_channel),
            ops.hwc2chw()
        ])
    else:
        print("Default normalization")
        op = ops.chain([
            ops.add(-mean),
            ops.mul(1 / std),
            ops.hwc2chw()
        ])

    print("Batch size: {}".format(args.batch_size))

    train_b = BatchGenerator(train_ds, args.batch_size, True, op)
    valid_b = BatchGenerator(valid_ds, args.batch_size, True, op)
    test_b = BatchGenerator(test_ds, args.batch_size, False, op)

    # check per channel normalization
    # b = next(iter(BatchGenerator(train_ds, len(train_ds), False, op)))
    # print(np.mean(b.data, axis=(0,2,3)))
    # print(np.std(b.data, axis=(0,2,3)))

    # basic recipe
    def conv_block(in_f, out_f, *args, **kwargs):
        return nn.Sequential(
                nn.Conv2d(in_f, out_f, *args, **kwargs),
                nn.ReLU(),
                nn.Conv2d(out_f, out_f, *args, **kwargs),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )

    class Net(nn.Module):
        def __init__(self, num_classes=2):
            super().__init__()
            self.enc_sizes = [3, 32, 64]
            conv_blocks = [conv_block(in_f, out_f, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                           for in_f, out_f in zip(self.enc_sizes, self.enc_sizes[1:])]

            self.encoder = nn.Sequential(*conv_blocks)
            self.decoder = nn.Sequential(
                nn.Linear(8 * 8 * 64, args.fc_size),
                nn.ReLU(),
                nn.Linear(args.fc_size, num_classes)
            )

        def forward(self, x):
            x = self.encoder(x)
            x = x.view(x.size(0), -1)
            x = self.decoder(x)
            return x

    net = Net()

    # visualize network with torchviz
    # from torchviz import make_dot
    # dummy_batch = next(iter(BatchGenerator(train_ds, 32, True, op)))
    # data = (torch.from_numpy(dummy_batch.data)).float()
    # yhat = net(data)
    # make_dot(yhat, params=dict(list(net.named_parameters()))).render("basic_recipe", format="png")

    # visualize with hidden layer
    # import hiddenlayer as hl
    # im = hl.build_graph(net, torch.zeros([1, 3, 32, 32]))
    # im.save(path="basic_recipe", format="png")
    print(net)

    if torch.cuda.is_available():
        net = net.cuda()

    print("== Training from scratch ===")
    clf = CnnClassifier(net, (0, 3, 32, 32), 2, lr=args.learning_rate, wd=0)

    start = datetime.now()
    for epoch in range(args.epoch):
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
    end = datetime.now()

    print("=== After full training: === ")
    acc = Accuracy()
    for batch in iter(train_b):
        acc.update(clf.predict(batch.data), batch.label)
    print("\ttrain acc: accuracy: {:.3f}".format(acc.accuracy()))
    train_acc = acc.accuracy()
    acc.reset()
    for batch in iter(valid_b):
        acc.update(clf.predict(batch.data), batch.label)
    print("\tval acc: accuracy: {:.3f}".format(acc.accuracy()))
    valid_acc = acc.accuracy()
    acc.reset()
    for batch in iter(test_b):
        acc.update(clf.predict(batch.data), batch.label)
    print("\ttest acc: accuracy: {:.3f}".format(acc.accuracy()))
    test_acc = acc.accuracy()

    out = {}
    out['time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    out['training_time'] = str(end-start)
    out['batch_size'] = args.batch_size
    out['channel_normalization'] = args.channel
    out['learning_rate'] = args.learning_rate
    out['last_layer_size'] = args.fc_size
    out['epochs'] = epoch+1
    out['train_loss_mean'] = losses.mean()
    out['train_loss_std'] = losses.std()
    out['train_acc'] = train_acc
    out['valid_acc'] = valid_acc
    out['test_acc'] = test_acc
    with open("result_part2", "a") as outfile:
        json.dump(out, outfile)
        outfile.write(',\n')



