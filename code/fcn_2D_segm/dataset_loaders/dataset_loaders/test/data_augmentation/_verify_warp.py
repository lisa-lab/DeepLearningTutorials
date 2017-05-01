import copy
import matplotlib.pyplot as plt
import numpy as np
import scipy
import SimpleITK as sitk

from data_augmentation import gen_warp_field
from data_augmentation import apply_warp as apply_warp_fra, pad_image


def apply_warp(x, warp_field, fill_mode='reflect',
               interpolator=sitk.sitkLinear,
               fill_constant=0):
    # Expand deformation field (and later the image), padding for the largest
    # deformation
    warp_field_arr = sitk.GetArrayFromImage(warp_field)
    max_deformation = np.max(np.abs(warp_field_arr))
    pad = np.ceil(max_deformation).astype(np.int32)
    warp_field_padded_arr = pad_image(warp_field_arr, pad_amount=pad,
                                      mode='nearest')
    warp_field_padded = sitk.GetImageFromArray(warp_field_padded_arr,
                                               isVector=True)

    # Warp x, one filter slice at a time
    x_warped = np.zeros(x.shape, dtype=np.float32)
    warp_filter = sitk.WarpImageFilter()
    warp_filter.SetInterpolator(interpolator)
    warp_filter.SetEdgePaddingValue(np.min(x).astype(np.double))
    for i, image in enumerate(x):
        x_tmp = np.zeros(image.shape, dtype=image.dtype)
        for j, channel in enumerate(image):
            image_padded = pad_image(channel, pad_amount=pad, mode=fill_mode,
                                     constant=fill_constant).T
            image_f = sitk.GetImageFromArray(image_padded)
            image_f_warped = warp_filter.Execute(image_f, warp_field_padded)
            image_warped = sitk.GetArrayFromImage(image_f_warped)
            x_tmp[j] = image_warped[pad:-pad, pad:-pad].T
        x_warped[i] = x_tmp
    return x_warped


def warp_michal(x, warp_field):
    x = apply_warp(x, warp_field, interpolator=sitk.sitkLinear,
                   fill_mode='constant', fill_constant=0)
    return x


def warp_fra(x, warp_field):
    x = apply_warp_fra(x, warp_field,
                       interpolator=sitk.sitkLinear,
                       fill_mode='constant',
                       fill_constant=0,
                       rows_idx=2, cols_idx=3)
    return x


if __name__ == '__main__':
    face = scipy.misc.face()
    face = face[None, ...]  # b01c
    face = face / 255.

    # Show
    def show(img, title=''):
        plt.imshow(img[0])
        plt.title(title)
        plt.show()

    # Michal bc01 assumption
    face = face.transpose((0, 3, 1, 2))

    # Warp
    warp_sigma = 80
    warp_grid_size = 5
    warp_field = gen_warp_field(shape=face.shape[-2:], sigma=warp_sigma,
                                grid_size=warp_grid_size)
    x_fra = warp_fra(copy.deepcopy(face), warp_field)
    x_michal = warp_michal(copy.deepcopy(face), warp_field)

    # Go back to b01c
    x_fra = x_fra.transpose((0, 2, 3, 1))
    x_michal = x_michal.transpose((0, 2, 3, 1))
    show(x_fra, 'fra')
    show(x_michal, 'michal')
    assert np.array_equal(x_fra, x_michal)
