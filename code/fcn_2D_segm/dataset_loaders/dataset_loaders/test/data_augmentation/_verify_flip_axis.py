import numpy as np


def flip_axis(x_in, axis):
    x_out = np.zeros(x_in.shape, dtype=x_in.dtype)
    for i, x in enumerate(x_in):
        x = np.asarray(x).swapaxes(axis, 0)
        x = x[::-1, ...]
        x_out[i] = x.swapaxes(0, axis)
    return x_out


def flip_axis_fra(x, flipping_axis):
    pattern = [flipping_axis]
    pattern += [el for el in range(x.ndim) if el != flipping_axis]
    inv_pattern = [pattern.index(el) for el in range(x.ndim)]
    x = x.transpose(pattern)  # "flipping_axis" first
    x = x[::-1, ...]
    x = x.transpose(inv_pattern)
    return x


if __name__ == '__main__':
    aa = np.random.random((10, 2, 3, 4))  # b, *, *, *
    for axis in [1, 2, 3]:
        print('Testing channel in axis {}'.format(axis))
        mm = flip_axis(aa.copy(), axis-1)
        ff = flip_axis_fra(aa.copy(), axis)
        assert np.array_equal(mm, ff)
    print('Test passed!')
