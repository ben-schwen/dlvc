import cv2
import matplotlib.pyplot as plt

from dlvc.batches import BatchGenerator, Batch
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
from torch.autograd import Variable
import torchvision
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

if __name__ == '__main__':
    # Parse args
    parser = argparse.ArgumentParser(description='Lorem ipsum')
    # data preparation
    parser.add_argument('--channel', action='store_true', help='Use per channel standardization')
    parser.add_argument('--hflip', action='store_true', help='Use horizontal flips for data augmentation')
    parser.add_argument('--rcrop', action='store_true', help='Use random cropping for data augmentation')
    parser.add_argument('--rerase', action='store_true', help='Use random erasing for data augmentation')
    parser.add_argument('--cutout', action='store_true', help='Use cutout for data augmentation')
    # non transfer learning
    parser.add_argument('--fc_size', type=int, default=1024, help='Size of last linear layer')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout for linear layer')
    # transfer learning
    parser.add_argument('--transfer', action='store_true', help='Use transfer learning')
    parser.add_argument('--feature', action='store_true', help='Use feature extractor for transfer learning')
    # allgemein
    parser.add_argument('--epoch', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--wd', type=float, default=1e-6, help='Weight decay')
    args = parser.parse_args()

    # 2. Create a BatchGenerator for each one. Traditional classifiers don't usually train in batches so you can set the
    # minibatch size equal to the number of dataset samples to get a single large batch - unless you choose a classifier
    # that does require multiple batches.
    ops_list = []
    if args.channel:
        ops_list += [ops.add(-mean_channel), ops.mul(1/std_channel)]
    else:
        ops_list += [ops.add(-mean), ops.mul(1/std)]

    ops_val = ops.chain(ops_list + [ops.hwc2chw()])

    if args.hflip:
        ops_list += ops.hflip()

    if args.rcrop:
        ops.rcrop(32, 4, pad_mode='reflect')

    if args.rerase:
        ops.rerase(p=1.0, sl=0, sh=0.4)

    if args.cutout:
        ops.cutout(length=8, p=0.5, n_holes=1)

    ops_train = ops.chain(ops_list + [ops.hwc2chw()])

    # ops_train = ops.chain([
    #     ops.add(-mean),
    #     ops.mul(1 / std),
    #     ops.hflip(),
    #     #ops.rotate90(),
    #     ops.rcrop(32, 4, pad_mode='reflect'),
    #     ops.rerase(p=1.0, sl=0, sh=0.4),
    #     # ops.cutout(length=8, p=0.5, n_holes=1),
    #     ops.hwc2chw()
    # ])

    # b = next(iter(BatchGenerator(train_ds, len(train_ds), False, op)))
    # print(np.mean(b.data, axis=(0,2,3)))
    # print(np.std(b.data, axis=(0,2,3)))

    print("Batch size: {}".format(args.batch_size))

    train_b = BatchGenerator(train_ds, args.batch_size, True, ops_train)
    valid_b = BatchGenerator(valid_ds, args.batch_size, True, ops_val)
    test_b = BatchGenerator(test_ds, args.batch_size, False, ops_val)


    # batch = next(iter(train_b))
    #
    # sample = batch.data[4]
    # plt.imshow(np.transpose(sample, (1,2,0)))
    # plt.show()

    # basic recipe
    def conv_block(in_f, out_f, p=0, *args, **kwargs):
        return nn.Sequential(
                nn.Conv2d(in_f, out_f, *args, **kwargs),
                nn.BatchNorm2d(out_f),
                nn.ReLU(),
                nn.Conv2d(out_f, out_f, *args, **kwargs),
                nn.BatchNorm2d(out_f),
                nn.ReLU(),
                nn.Conv2d(out_f, out_f, *args, **kwargs),
                nn.BatchNorm2d(out_f),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
                nn.Dropout(p)
        )

    # best so far, has 2 convolution blocks 32,64,128, Linear layer with 512 and Dropout=0.5 -> 0.89 valid
    class Net(nn.Module):
        def __init__(self, num_classes=2):
            super().__init__()
            self.enc_sizes = [3, 32, 64, 128]
            self.dropout_sizes = [0, 0, 0, 0]

            conv_blocks = [conv_block(in_f, out_f, p, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                           for in_f, out_f, p in zip(self.enc_sizes, self.enc_sizes[1:], self.dropout_sizes)]

            self.encoder = nn.Sequential(*conv_blocks)
            self.decoder = nn.Sequential(
                nn.Linear(4 * 4 * 128, args.fc_size),
                nn.ReLU(),
                nn.BatchNorm1d(args.fc_size),
                nn.Dropout(p=args.dropout),
                nn.Linear(args.fc_size, num_classes)
            )

        def forward(self, x):
            x = self.encoder(x)
            x = x.view(x.size(0), -1)
            x = self.decoder(x)
            return x

    net = Net()

    # #visualize with hidden layer

    # import hiddenlayer as hl
    # im = hl.build_graph(net, torch.zeros([16, 3, 32, 32]))
    # im.save(path="part3_net", format="png")
    print(net)

    if torch.cuda.is_available():
        net = net.cuda()

    clf = CnnClassifier(net, (0, 3, 32, 32), 2, lr=args.learning_rate, wd=args.wd)

    def set_parameter_requires_grad(model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False

    if args.transfer:
        print("== Transfer learning ==")
        net = models.resnet18(pretrained=True)
        set_parameter_requires_grad(net, args.feature_extract)
        num_ftrs = net.fc.in_features
        num_classes = 2
        net.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes))

        if torch.cuda.is_available():
            net = net.cuda()

        clf = CnnClassifier(net, (0, 3, 32, 32), 2, lr=args.learning_rate, wd=args.wd, size=224)
    else:
        print("== Training from scratch ===")

    best_acc = 0
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
        if acc.accuracy() > best_acc:
            best_acc = acc.accuracy()
            torch.save(clf.net, 'best_model.pt')

    best_model = torch.load('best_model.pt')
    if torch.cuda.is_available():
        best_model = best_model.cuda()
    clf.net = best_model

    start = datetime.now()
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
    end = datetime.now()

    out = {}
    # time
    out['time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    out['training_time'] = str(end - start)
    # data augmentation
    out['channel_normalization'] = args.channel
    out['horizontal_flip'] = args.hflip
    out['random_cropping'] = args.rcrop
    out['random_erasing'] = args.rerase
    out['cutout'] = args.cutout
    # non transfer learning
    out['last_layer_size'] = args.fc_size
    out['dropout'] = args.dropout
    # transfer learning
    out['transfer_learning'] = args.transfer
    out['feature_extracting'] = args.feature
    # allgemein
    out['epochs'] = epoch + 1
    out['batch_size'] = args.batch_size
    out['learning_rate'] = args.learning_rate
    out['weight_decay'] = args.wd
    # metrics
    out['train_loss_mean'] = losses.mean()
    out['train_loss_std'] = losses.std()
    out['train_acc'] = train_acc
    out['valid_acc'] = valid_acc
    out['test_acc'] = test_acc
    with open("result_part3", "a") as outfile:
        json.dump(out, outfile)
        outfile.write(',\n')

# print("Transfer learning")
# def set_parameter_requires_grad(model, feature_extracting):
#     if feature_extracting:
#         for param in model.parameters():
#             param.requires_grad = False
#
# # finetuning
# net = models.resnet18(pretrained=True)
# set_parameter_requires_grad(net, False)
#
# num_ftrs = net.fc.in_features
# num_classes = 2
# net.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes))
#
# if torch.cuda.is_available():
#     net = net.cuda()
#
# clf = CnnClassifier(net, (0, 3, 32, 32), 2, lr=0.001, wd=0.01, size=224)
#
# best_acc = 0
# for epoch in range(60):
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
#     if acc.accuracy() > best_acc:
#         best_acc = acc.accuracy()
#         torch.save(clf.net, 'best_model.pt')
#
# best_model = torch.load('best_model.pt')
# if torch.cuda.is_available():
#     best_model = best_model.cuda()
# clf.net = best_model
#
# print("=== After full training: === ")
# acc = Accuracy()
# for batch in iter(train_b):
#     acc.update(clf.predict(batch.data), batch.label)
# print("\ttrain acc: accuracy: {:.3f}".format(acc.accuracy()))
# acc.reset()
# for batch in iter(valid_b):
#     acc.update(clf.predict(batch.data), batch.label)
# print("\tval acc: accuracy: {:.3f}".format(acc.accuracy()))
# acc.reset()
# for batch in iter(test_b):
#     acc.update(clf.predict(batch.data), batch.label)
# print("\ttest acc: accuracy: {:.3f}".format(acc.accuracy()))