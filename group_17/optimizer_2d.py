import os
from collections import namedtuple

import cv2
import numpy as np
import torch

# A 2D vector. Used in Fn as an evaluation point.
Vec2 = namedtuple('Vec2', ['x1', 'x2'])

class AutogradFn(torch.autograd.Function):
    '''
    This class wraps a Fn instance to make it compatible with PyTorch optimizers
    '''
    @staticmethod
    def forward(ctx, fn, loc):
        ctx.fn = fn
        ctx.save_for_backward(loc)
        value = fn(Vec2(loc[0].item(), loc[1].item()))
        return torch.tensor(value)

    @staticmethod
    def backward(ctx, grad_output):
        fn = ctx.fn
        loc, = ctx.saved_tensors
        grad = fn.grad(Vec2(loc[0].item(), loc[1].item()))
        return None, torch.tensor([grad.x1, grad.x2]) * grad_output


def load_image(fpath: str) -> np.ndarray:
    '''
    Loads a 2D function from a PNG file and normalizes it to the interval [0, 1]
    Raises FileNotFoundError if the file does not exist.
    '''

    # TODO implement
    if not os.path.isfile(fpath):
        raise FileNotFoundError()

    img = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
    img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return img

class Fn:
    '''
    A 2D function evaluated on a grid.
    '''

    def __init__(self, fn: np.ndarray, eps: float):
        '''
        Ctor that assigns function data fn and step size eps for numerical differentiation
        '''

        self.fn = fn
        self.eps = eps

    def visualize(self) -> np.ndarray:
        '''
        Return a visualization of the function as a color image. Use e.g. cv2.applyColorMap.
        Use the result to visualize the progress of gradient descent.
        '''

        # TODO implement
        img_grey = self.fn
        img = img_grey * 255
        img = img.astype(np.uint8)
        img_color = cv2.applyColorMap(img, cv2.COLORMAP_JET)

        return img_color

    def __call__(self, loc: Vec2) -> float:
        '''
        Evaluate the function at location loc.
        Raises ValueError if loc is out of bounds.
        '''

        # TODO implement
        # You can simply round and map to integers. If so, make sure not to set eps and learning_rate too low
        # Alternatively, you can implement some form of interpolation (for example bilinear)
        if loc.x1 < 0 or loc.x2 < 0:
            raise ValueError('Index out of bounds')
        if loc.x1 >= self.fn.shape[0] or loc.x2 >= self.fn.shape[1]:
            raise ValueError('Index out of bounds')

        # cuz of type fuckup and mixing of named tuple, tensor and numpy...

        x = loc[0]
        y = loc[1]

        if torch.is_tensor(x):
            x = x.detach().numpy()
        if torch.is_tensor(y):
            y = y.detach().numpy()

        x1 = np.floor(x).astype(int)
        x2 = np.floor(x + 1).astype(int)
        y1 = np.floor(y).astype(int)
        y2 = np.floor(y + 1).astype(int)

        q11 = self.fn[x1, y1]
        q12 = self.fn[x1, y2]
        q21 = self.fn[x2, y1]
        q22 = self.fn[x2, y2]

        return (1 / ((x2 - x1) * (y2 - y1)) * np.array([[x2 - x, x - x1]]) @ np.array([[q11, q12], [q21, q22]]) @ \
               np.array([[y2 - y], [y - y1]]))[0][0]

        #return self.fn[tuple(np.round(loc).astype('int'))]

    def grad(self, loc: Vec2) -> Vec2:
        '''
        Compute the numerical gradient of the function at location loc, using the given epsilon.
        Raises ValueError if loc is out of bounds of fn or if eps <= 0.
        '''

        # TODO implement
        if self.eps <= 0:
            raise ValueError('eps =< 0.')
        # double the code because of bad requirements
        if loc.x1 < 0 or loc.x2 < 0:
            raise ValueError('Index out of bounds')
        if loc.x1 >= self.fn.shape[0] or loc.x2 >= self.fn.shape[1]:
            raise ValueError('Index out of bounds')

        x1 = (fn(Vec2(loc.x1 + self.eps, loc.x2)) - fn(Vec2(loc.x1 - self.eps, loc.x2))) / (2 * self.eps)
        x2 = (fn(Vec2(loc.x1, loc.x2 + self.eps)) - fn(Vec2(loc.x1, loc.x2 - self.eps))) / (2 * self.eps)

        return Vec2(x1, x2)



if __name__ == '__main__':
    # Parse args
    import argparse

    parser = argparse.ArgumentParser(description='Perform gradient descent on a 2D function.')
    parser.add_argument('fpath', help='Path to a PNG file encoding the function')
    parser.add_argument('sx1', type=float, help='Initial value of the first argument')
    parser.add_argument('sx2', type=float, help='Initial value of the second argument')
    parser.add_argument('--eps', type=float, default=1.0, help='Epsilon for computing numeric gradients')
    parser.add_argument('--learning_rate', type=float, default=10.0, help='Learning rate')
    parser.add_argument('--beta', type=float, default=0, help='Beta parameter of momentum (0 = no momentum)')
    parser.add_argument('--nesterov', action='store_true', help='Use Nesterov momentum')
    parser.add_argument('--rms', action='store_true', help='Use AdamW optimizer instead of SGD')
    args = parser.parse_args()

    # REPL
    # class Object(object):
    #     pass
    # args = Object()
    # args.fpath = '/home/bschwendinger/github/dlvc/group_17/dlvc/fn/beale.png'
    # args.sx1 = 300.0
    # args.sx2 = 300.0
    # args.eps = 1.0
    # args.learning_rate = 3000
    # args.beta = 0
    # args.nesterov = False

    # Init
    image_fn = load_image(args.fpath)
    fn = Fn(image_fn, args.eps)
    vis = fn.visualize()

    # PyTorch uses tensors which are very similar to numpy arrays but hold additional values such as gradients
    loc = torch.tensor([args.sx1, args.sx2], requires_grad=True)
    optimizer = torch.optim.SGD([loc], lr=args.learning_rate, momentum=args.beta, nesterov=args.nesterov)

    if args.rms:
        optimizer = torch.optim.RMSprop([loc], lr=args.learning_rate, momentum=args.beta)

    # Find a minimum in fn using a PyTorch optimizer
    # See https://pytorch.org/docs/stable/optim.html for how to use optimizers
    epoch = 0
    epoch_max = 1000
    points = []
    thresh = 1e-10
    while True:
        # Visualize each iteration by drawing on vis using e.g. cv2.line()
        # Find a suitable termination condition and break out of loop once done
        # This returns the value of the function fn at location loc.
        # Since we are trying to find a minimum of the function this acts as a loss value.
        # loss = AutogradFn.apply(fn, loc)
        epoch += 1
        start = tuple(np.rint(loc.detach().numpy()).astype('uint16'))

        optimizer.zero_grad()
        gradient = fn.grad(Vec2(loc.data[0], loc.data[1]))
        loss = AutogradFn.apply(fn, loc)
        loss.backward()
        optimizer.step()

        end = tuple(np.rint(loc.detach().numpy()).astype('uint16'))

        if epoch >= epoch_max or np.max(np.abs(gradient)) < thresh:
            print("Number of max epochs reached or gradient became too small")
            print("Epochs: {} of maximum {} epochs".format(epoch, epoch_max))
            print("Gradient: {}".format(gradient))
            print("Minimum at: {}".format(loc.detach().numpy()))
            print("Minimum value: {}".format(fn(Vec2(loc[0].item(), loc[1].item()))))
            break

        cv2.line(vis, start, end, color=[0,0,255], thickness=3)
        cv2.imshow('Progress', vis)
        cv2.waitKey(10)  # 20 fps, tune according to your liking

# python optimizer_2d.py "/home/bschwendinger/github/dlvc/group_17/dlvc/fn/beale.png" 300 300 --learning_rate 10000 --beta 0.96 --nesterov