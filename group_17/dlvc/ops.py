from typing import List, Callable

import numpy as np

# All operations are functions that take and return numpy arrays
# See https://docs.python.org/3/library/typing.html#typing.Callable for what this line means
Op = Callable[[np.ndarray], np.ndarray]

def chain(ops: List[Op]) -> Op:
    '''
    Chain a list of operations together.
    '''

    def op(sample: np.ndarray) -> np.ndarray:
        for op_ in ops:
            sample = op_(sample)
        return sample

    return op

def type_cast(dtype: np.dtype) -> Op:
    '''
    Cast numpy arrays to the given type.
    '''

    # TODO implement (see above for guidance).
    return lambda x: x.astype(dtype)

def vectorize() -> Op:
    '''
    Vectorize numpy arrays via "numpy.ravel()".
    '''

    # TODO implement (see above for guidance).
    return lambda x: np.ravel(x)
    # return lambda x: x.reshape(-1,  np.prod(np.shape(x)[1:]))

def hwc2chw() -> Op:
    '''
    Flip a 3D array with shape HWC to shape CHW.
    '''
    # TODO implement (see np.transpose)
    # return lambda x: np.einsum('hwc->chw', x)
    return lambda x: np.transpose(x, (2, 0, 1))

def add(val: float) -> Op:
    '''
    Add a scalar value to all array elements.
    '''

    # TODO implement (see above for guidance).
    return lambda x: x + val

def mul(val: float) -> Op:
    '''
    Multiply all array elements by the given scalar.
    '''

    # TODO implement (see above for guidance).
    return lambda x: x * val

def hflip() -> Op:
    '''
    Flip arrays with shape HWC horizontally with a probability of 0.5.
    '''

    # TODO implement (numpy.flip will be helpful)
    def op(sample: np.ndarray) -> np.ndarray:
        if np.random.random() < 0.5:
            return np.flip(sample, axis=1)
        return sample

    return op

def rcrop(sz: int, pad: int, pad_mode: str) -> Op:
    '''
    Extract a square random crop of size sz from arrays with shape HWC.
    If pad is > 0, the array is first padded by pad pixels along the top, left, bottom, and right.
    How padding is done is governed by pad_mode, which should work exactly as the 'mode' argument of numpy.pad.
    Raises ValueError if sz exceeds the array width/height after padding.
    '''

    # TODO implement
    # https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.pad.html will be helpful
    # https://pytorch.org/vision/stable/_modules/torchvision/transforms/transforms.html#RandomCrop is more helpful
    def op(sample: np.ndarray) -> np.ndarray:
        if pad > 0:
            sample = np.pad(sample, ((pad, pad), (pad, pad), (0, 0)), pad_mode)
        h, w, _ = sample.shape
        if sz > h or sz > w:
            raise ValueError("Required crop size {} is larger then padded input image size {}".format((sz, sz), (h, w)))

        i = np.random.randint(0, h-sz)
        j = np.random.randint(0, w-sz)
        return sample[i:(i+sz), j:(j+sz), :]

    return op

def vflip() -> Op:
    '''
    Flip arrays with shape HWC vertically with a probability of 0.5.
    '''

    # TODO implement (numpy.flip will be helpful)
    def op(sample: np.ndarray) -> np.ndarray:
        if np.random.random() < 0.5:
            return np.flip(sample, axis=2)
        return sample

    return op

def rotate90() -> Op:
    '''
    Rotate arrays with shape HWC left or right 90% with a probability of 0.25 each.
    '''

    # TODO implement (numpy.flip will be helpful)
    def op(sample: np.ndarray) -> np.ndarray:
        if np.random.random() < 0.5:
            if np.random.random() < 0.5:
                return np.rot90(sample, k=1, axes=(0, 1))
            else:
                return np.rot90(sample, k=3, axes=(0, 1))
        return sample

    return op

def rerase(p=.5, sl=0.02, sh=0.4, r1=0.3, pixel_level=True) -> Op:
    """
    p: probability that erase operation takes place
    sl: min erased area
    sh: max erased area
    r1: min aspect ratio
    pixel_level: erased area is replaced pixel by pixel or with a single color
    """
    def op(sample: np.ndarray) -> np.ndarray:
        if np.random.uniform(0, 1) > p:
            return sample

        img_h, img_w, img_c = sample.shape

        for attempt in range(100):
            s = np.random.uniform(sl, sh) * img_h * img_w
            r = np.random.uniform(r1, 1/r1)

            h = int(np.sqrt(s * r))
            w = int(np.sqrt(s / r))

            if w < img_w and h < img_h:
                left = np.random.randint(0, img_w - w)
                top = np.random.randint(0, img_h - h)
                if pixel_level:
                    c = np.random.uniform(-1, 1, (h, w, img_c))
                else:
                    c = np.random.uniform(-1, 1)
                sample[top:top+h, left:left+w] = c
                break
        return sample
    return op