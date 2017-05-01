import numpy as np


def random_channel_shift(x_in, intensity, channel_index=0):
    x_out = np.zeros(x_in.shape, dtype=x_in.dtype)
    for i, x in enumerate(x_in):
        x = np.rollaxis(x, channel_index, 0)
        min_x, max_x = np.min(x), np.max(x)
        # channel_images = [np.clip(x_channel + np.random.uniform(-intensity,
        #                                                         intensity),
        #                           min_x, max_x)
        channel_images = [np.clip(x_channel, min_x, max_x) for x_channel in x]
        x = np.stack(channel_images, axis=0)
        x_out[i] = np.rollaxis(x, 0, channel_index+1)
    return x_out


def random_channel_shift_fra(x, intensity, channel_index=0):
    pattern = [channel_index]
    pattern += [el for el in range(x.ndim) if el != channel_index]
    inv_pattern = [pattern.index(el) for el in range(x.ndim)]
    x = x.transpose(pattern)  # channel first
    x_shape = list(x.shape)
    x = x.reshape((x_shape[0], -1))  # squash everything else on last axis
    # Loop on channels
    for i in range(x.shape[0]):
        min_x, max_x = np.min(x), np.max(x)
        # x[i] = np.clip(x[i], + np.random.uniform(-intensity, intensity),
        #                min_x, max_x)
        x[i] = np.clip(x[i], min_x, max_x)
    x = x.reshape(x_shape)  # unsquash
    x = x.transpose(inv_pattern)
    return x


if __name__ == '__main__':
    aa = np.random.random((10, 2, 3, 4))  # b, *, *, *
    for axis in [1, 2, 3]:
        print('Testing channel in axis {}'.format(axis))
        mm = random_channel_shift(aa.copy(), 2, axis - 1)
        ff = random_channel_shift_fra(aa.copy(), 2, axis)
        assert np.allclose(mm, ff)
    print('Test passed!')
