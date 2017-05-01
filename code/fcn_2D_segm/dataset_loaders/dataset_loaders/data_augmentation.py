# Based on
# https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py
import os

import numpy as np
from scipy import interpolate
import scipy.misc
import scipy.ndimage as ndi
from skimage.color import rgb2gray, gray2rgb
from skimage import img_as_float


def optical_flow(seq, rows_idx, cols_idx, chan_idx, return_rgb=False):
    '''Optical flow

    Takes a 4D array of sequences and returns a 4D array with
    an RGB optical flow image for each frame in the input'''
    import cv2
    if seq.ndim != 4:
        raise RuntimeError('Optical flow expected 4 dimensions, got %d' %
                           seq.ndim)
    seq = seq.copy()
    seq = (seq * 255).astype('uint8')
    # Reshape to channel last: (b*seq, 0, 1, ch) if seq
    pattern = [el for el in range(seq.ndim)
               if el not in (rows_idx, cols_idx, chan_idx)]
    pattern += [rows_idx, cols_idx, chan_idx]
    inv_pattern = [pattern.index(el) for el in range(seq.ndim)]
    seq = seq.transpose(pattern)
    if seq.shape[0] == 1:
        raise RuntimeError('Optical flow needs a sequence longer than 1 '
                           'to work')
    seq = seq[..., ::-1]  # Go BGR for OpenCV

    frame1 = seq[0]
    if return_rgb:
        flow_seq = np.zeros_like(seq)
        hsv = np.zeros_like(frame1)
    else:
        sh = list(seq.shape)
        sh[-1] = 2
        flow_seq = np.zeros(sh)

    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)  # Go to gray

    flow = None
    for i, frame2 in enumerate(seq[1:]):
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)  # Go to gray
        flow = cv2.calcOpticalFlowFarneback(
            frame1, frame2, 0.5, 3, 10, 3, 5, 1.1, 0, flow)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        if return_rgb:
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 1] = 255
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            rgb = bgr[:, :, ::-1]
            flow_seq[i+1] = rgb
            # Image.fromarray(rgb).show()
            # cv2.imwrite('opticalfb.png', frame2)
            # cv2.imwrite('opticalhsv.png', bgr)
        else:
            flow_seq[i+1] = np.stack((mag, ang), 2)
        frame1 = frame2
    flow_seq = flow_seq.transpose(inv_pattern)
    return flow_seq / 255.  # return in [0, 1]


def my_label2rgb(labels, cmap, bglabel=None, bg_color=(0., 0., 0.)):
    '''Convert a label mask to RGB applying a color map'''
    output = np.zeros(labels.shape + (3,), dtype=np.float64)
    for i in range(len(cmap)):
        if i != bglabel:
            output[(labels == i).nonzero()] = cmap[i]
    if bglabel is not None:
        output[(labels == bglabel).nonzero()] = bg_color
    return output


def my_label2rgboverlay(labels, cmap, image, bglabel=None,
                        bg_color=(0., 0., 0.), alpha=0.2):
    '''Superimpose a mask over an image

    Convert a label mask to RGB applying a color map and superimposing it
    over an image as a transparent overlay'''
    image_float = gray2rgb(img_as_float(rgb2gray(image)))
    label_image = my_label2rgb(labels, cmap, bglabel=bglabel,
                               bg_color=bg_color)
    output = image_float * alpha + label_image * (1 - alpha)
    return output


def save_img2(x, y, fname, cmap, void_label, rows_idx, cols_idx,
              chan_idx):
    '''Save a mask and an image side to side

    Convert a label mask to RGB applying a color map and superimposing it
    over an image as a transparent overlay. Saves the original image and
    the image with the mask overlay in a file'''
    pattern = [el for el in range(x.ndim) if el not in [rows_idx, cols_idx,
                                                        chan_idx]]
    pattern += [rows_idx, cols_idx, chan_idx]
    x_copy = x.transpose(pattern)
    if y is not None and len(y) > 0:
        y_copy = y.transpose(pattern)

    # Take the first batch and drop extra dim on y
    x_copy = x_copy[0]
    if y is not None and len(y) > 0:
        y_copy = y_copy[0, ..., 0]

    label_mask = my_label2rgboverlay(y,
                                     colors=cmap,
                                     image=x,
                                     bglabel=void_label,
                                     alpha=0.2)
    combined_image = np.concatenate((x, label_mask),
                                    axis=1)
    scipy.misc.toimage(combined_image).save(fname)


def transform_matrix_offset_center(matrix, x, y):
    '''Shift the transformation matrix to be in the center of the image

    Apply an offset to the transformation matrix so that the origin of
    the axis is in the center of the image.'''
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def apply_transform(x, transform_matrix, fill_mode='nearest', cval=0.,
                    order=0, rows_idx=1, cols_idx=2):
    '''Apply an affine transformation on each channel separately.'''
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]

    # Reshape to (*, 0, 1)
    pattern = [el for el in range(x.ndim) if el != rows_idx and el != cols_idx]
    pattern += [rows_idx, cols_idx]
    inv_pattern = [pattern.index(el) for el in range(x.ndim)]
    x = x.transpose(pattern)
    x_shape = list(x.shape)
    x = x.reshape([-1] + x_shape[-2:])  # squash everything on the first axis

    # Apply the transformation on each channel, sequence, batch, ..
    for i in range(x.shape[0]):
        x[i] = ndi.interpolation.affine_transform(x[i], final_affine_matrix,
                                                  final_offset, order=order,
                                                  mode=fill_mode, cval=cval)
    x = x.reshape(x_shape)  # unsquash
    x = x.transpose(inv_pattern)
    return x


def random_channel_shift(x, shift_range, rows_idx, cols_idx, chan_idx):
    '''Shift the intensity values of each channel uniformly.

    Channel by channel, shift all the intensity values by a random value in
    [-shift_range, shift_range]'''
    pattern = [chan_idx]
    pattern += [el for el in range(x.ndim) if el not in [rows_idx, cols_idx,
                                                         chan_idx]]
    pattern += [rows_idx, cols_idx]
    inv_pattern = [pattern.index(el) for el in range(x.ndim)]
    x = x.transpose(pattern)  # channel first
    x_shape = list(x.shape)
    # squash rows and cols together and everything else on the 1st
    x = x.reshape((-1, x_shape[-2] * x_shape[-1]))
    # Loop on the channels/batches/etc
    for i in range(x.shape[0]):
        min_x, max_x = np.min(x), np.max(x)
        x[i] = np.clip(x[i] + np.random.uniform(-shift_range, shift_range),
                       min_x, max_x)
    x = x.reshape(x_shape)  # unsquash
    x = x.transpose(inv_pattern)
    return x


def flip_axis(x, flipping_axis):
    '''Flip an axis by inverting the position of its elements'''
    pattern = [flipping_axis]
    pattern += [el for el in range(x.ndim) if el != flipping_axis]
    inv_pattern = [pattern.index(el) for el in range(x.ndim)]
    x = x.transpose(pattern)  # "flipping_axis" first
    x = x[::-1, ...]
    x = x.transpose(inv_pattern)
    return x


def pad_image(x, pad_amount, mode='reflect', constant=0.):
    '''Pad an image

    Pad an image by pad_amount on each side.

    Parameters
    ----------
    x: numpy ndarray
        The array to be padded.
    pad_amount: int
        The number of pixels of the padding.
    mode: string
        The padding mode. If "constant" a constant value will be used to
        fill the padding; if "reflect" the border pixels will be used in
        inverse order to fill the padding; if "nearest" the border pixel
        closer to the padded area will be used to fill the padding; if
        "zero" the padding will be filled with zeros.
    constant: int
        The value used to fill the padding when "constant" mode is
        selected.
        '''
    e = pad_amount
    shape = list(x.shape)
    shape[:2] += 2*e
    if mode == 'constant':
        x_padded = np.ones(shape, dtype=np.float32)*constant
        x_padded[e:-e, e:-e] = x.copy()
    else:
        x_padded = np.zeros(shape, dtype=np.float32)
        x_padded[e:-e, e:-e] = x.copy()

    if mode == 'reflect':
        # Edges
        x_padded[:e, e:-e] = np.flipud(x[:e, :])  # left
        x_padded[-e:, e:-e] = np.flipud(x[-e:, :])  # right
        x_padded[e:-e, :e] = np.fliplr(x[:, :e])  # top
        x_padded[e:-e, -e:] = np.fliplr(x[:, -e:])  # bottom
        # Corners
        x_padded[:e, :e] = np.fliplr(np.flipud(x[:e, :e]))  # top-left
        x_padded[-e:, :e] = np.fliplr(np.flipud(x[-e:, :e]))  # top-right
        x_padded[:e, -e:] = np.fliplr(np.flipud(x[:e, -e:]))  # bottom-left
        x_padded[-e:, -e:] = np.fliplr(np.flipud(x[-e:, -e:]))  # bottom-right
    elif mode == 'zero' or mode == 'constant':
        pass
    elif mode == 'nearest':
        # Edges
        x_padded[:e, e:-e] = x[[0], :]  # left
        x_padded[-e:, e:-e] = x[[-1], :]  # right
        x_padded[e:-e, :e] = x[:, [0]]  # top
        x_padded[e:-e, -e:] = x[:, [-1]]  # bottom
        # Corners
        x_padded[:e, :e] = x[[0], [0]]  # top-left
        x_padded[-e:, :e] = x[[-1], [0]]  # top-right
        x_padded[:e, -e:] = x[[0], [-1]]  # bottom-left
        x_padded[-e:, -e:] = x[[-1], [-1]]  # bottom-right
    else:
        raise ValueError("Unsupported padding mode \"{}\"".format(mode))
    return x_padded


def gen_warp_field(shape, sigma=0.1, grid_size=3):
    '''Generate an spline warp field'''
    import SimpleITK as sitk
    # Initialize bspline transform
    args = shape+(sitk.sitkFloat32,)
    ref_image = sitk.Image(*args)
    tx = sitk.BSplineTransformInitializer(ref_image, [grid_size, grid_size])

    # Initialize shift in control points:
    # mesh size = number of control points - spline order
    p = sigma * np.random.randn(grid_size+3, grid_size+3, 2)

    # Anchor the edges of the image
    p[:, 0, :] = 0
    p[:, -1:, :] = 0
    p[0, :, :] = 0
    p[-1:, :, :] = 0

    # Set bspline transform parameters to the above shifts
    tx.SetParameters(p.flatten())

    # Compute deformation field
    displacement_filter = sitk.TransformToDisplacementFieldFilter()
    displacement_filter.SetReferenceImage(ref_image)
    displacement_field = displacement_filter.Execute(tx)

    return displacement_field


def apply_warp(x, warp_field, fill_mode='reflect',
               interpolator=None,
               fill_constant=0, rows_idx=1, cols_idx=2):
    '''Apply an spling warp field on an image'''
    import SimpleITK as sitk
    if interpolator is None:
        interpolator = sitk.sitkLinear
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
    pattern = [el for el in range(0, x.ndim) if el not in [rows_idx, cols_idx]]
    pattern += [rows_idx, cols_idx]
    inv_pattern = [pattern.index(el) for el in range(x.ndim)]
    x = x.transpose(pattern)  # batch, channel, ...
    x_shape = list(x.shape)
    x = x.reshape([-1] + x_shape[2:])  # *, r, c
    warp_filter = sitk.WarpImageFilter()
    warp_filter.SetInterpolator(interpolator)
    warp_filter.SetEdgePaddingValue(np.min(x).astype(np.double))
    for i in range(x.shape[0]):
        bc_pad = pad_image(x[i], pad_amount=pad, mode=fill_mode,
                           constant=fill_constant).T
        bc_f = sitk.GetImageFromArray(bc_pad)
        bc_f_warped = warp_filter.Execute(bc_f, warp_field_padded)
        bc_warped = sitk.GetArrayFromImage(bc_f_warped)
        x[i] = bc_warped[pad:-pad, pad:-pad].T
    x = x.reshape(x_shape)  # unsquash
    x = x.transpose(inv_pattern)
    return x


def random_transform(x, y=None,
                     rotation_range=0.,
                     width_shift_range=0.,
                     height_shift_range=0.,
                     shear_range=0.,
                     zoom_range=0.,
                     channel_shift_range=0.,
                     fill_mode='nearest',
                     cval=0.,
                     cvalMask=0.,
                     horizontal_flip=0.,  # probability
                     vertical_flip=0.,  # probability
                     rescale=None,
                     spline_warp=False,
                     warp_sigma=0.1,
                     warp_grid_size=3,
                     crop_size=None,
                     return_optical_flow=False,
                     nclasses=None,
                     gamma=0.,
                     gain=1.,
                     chan_idx=3,  # No batch yet: (s, 0, 1, c)
                     rows_idx=1,  # No batch yet: (s, 0, 1, c)
                     cols_idx=2,  # No batch yet: (s, 0, 1, c)
                     void_label=None):
    '''Random Transform.

    A function to perform data augmentation of images and masks during
    the training  (on-the-fly). Based on [1]_.

    Parameters
    ----------
    x: array of floats
        An image.
    y: array of int
        An array with labels.
    rotation_range: int
        Degrees of rotation (0 to 180).
    width_shift_range: float
        The maximum amount the image can be shifted horizontally (in
        percentage).
    height_shift_range: float
        The maximum amount the image can be shifted vertically (in
        percentage).
    shear_range: float
        The shear intensity (shear angle in radians).
    zoom_range: float or list of floats
        The amout of zoom. If set to a scalar z, the zoom range will be
        randomly picked in the range [1-z, 1+z].
    channel_shift_range: float
        The shift range for each channel.
    fill_mode: string
        Some transformations can return pixels that are outside of the
        boundaries of the original image. The points outside the
        boundaries are filled according to the given mode (`constant`,
        `nearest`, `reflect` or `wrap`). Default: `nearest`.
    cval: int
        Value used to fill the points of the image outside the boundaries when
        fill_mode is `constant`. Default: 0.
    cvalMask: int
        Value used to fill the points of the mask outside the boundaries when
        fill_mode is `constant`. Default: 0.
    horizontal_flip: float
        The probability to randomly flip the images (and masks)
        horizontally. Default: 0.
    vertical_flip: bool
        The probability to randomly flip the images (and masks)
        vertically. Default: 0.
    rescale: float
        The rescaling factor. If None or 0, no rescaling is applied, otherwise
        the data is multiplied by the value provided (before applying
        any other transformation).
    spline_warp: bool
        Whether to apply spline warping.
    warp_sigma: float
        The sigma of the gaussians used for spline warping.
    warp_grid_size: int
        The grid size of the spline warping.
    crop_size: tuple
        The size of crop to be applied to images and masks (after any
        other transformation).
    return_optical_flow: bool
        If not False a dense optical flow will be concatenated to the
        end of the channel axis of the image. If True, magnitude and
        angle will be returned, if set to 'rbg' an RGB representation
        will be returned instead. Default: False.
    nclasses: int
        The number of classes of the dataset.
    gamma: float
        Controls gamma in Gamma correction.
    gain: float
        Controls gain in Gamma correction.
    chan_idx: int
        The index of the channel axis.
    rows_idx: int
        The index of the rows of the image.
    cols_idx: int
        The index of the cols of the image.
    void_label: int
        The index of the void label, if any. Used for padding.

    Reference
    ---------
    [1] https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py
    '''
    # Set this to a dir, if you want to save augmented images samples
    save_to_dir = None

    if rescale:
        raise NotImplementedError()

    # Do not modify the original images
    x = x.copy()
    if y is not None and len(y) > 0:
        y = y[..., None]  # Add extra dim to y to simplify computation
        y = y.copy()

    # listify zoom range
    if np.isscalar(zoom_range):
        if zoom_range > 1.:
            raise RuntimeError('Zoom range should be between 0 and 1. '
                               'Received: ', zoom_range)
        zoom_range = [1 - zoom_range, 1 - zoom_range]
    elif len(zoom_range) == 2:
        if any(el > 1. for el in zoom_range):
            raise RuntimeError('Zoom range should be between 0 and 1. '
                               'Received: ', zoom_range)
        zoom_range = [1-el for el in zoom_range]
    else:
        raise Exception('zoom_range should be a float or '
                        'a tuple or list of two floats. '
                        'Received arg: ', zoom_range)

    # Channel shift
    if channel_shift_range != 0:
        x = random_channel_shift(x, channel_shift_range, rows_idx, cols_idx,
                                 chan_idx)

    # Gamma correction
    if gamma > 0:
        scale = float(1)
        x = ((x / scale) ** gamma) * scale * gain

    # Affine transformations (zoom, rotation, shift, ..)
    if (rotation_range or height_shift_range or width_shift_range or
            shear_range or zoom_range != [1, 1]):

        # --> Rotation
        if rotation_range:
            theta = np.pi / 180 * np.random.uniform(-rotation_range,
                                                    rotation_range)
        else:
            theta = 0
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])
        # --> Shift/Translation
        if height_shift_range:
            tx = (np.random.uniform(-height_shift_range, height_shift_range) *
                  x.shape[rows_idx])
        else:
            tx = 0
        if width_shift_range:
            ty = (np.random.uniform(-width_shift_range, width_shift_range) *
                  x.shape[cols_idx])
        else:
            ty = 0
        translation_matrix = np.array([[1, 0, tx],
                                       [0, 1, ty],
                                       [0, 0, 1]])
        # --> Shear
        if shear_range:
            shear = np.random.uniform(-shear_range, shear_range)
        else:
            shear = 0
        shear_matrix = np.array([[1, -np.sin(shear), 0],
                                 [0, np.cos(shear), 0],
                                 [0, 0, 1]])
        # --> Zoom
        if zoom_range == [1, 1]:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
        zoom_matrix = np.array([[zx, 0, 0],
                                [0, zy, 0],
                                [0, 0, 1]])
        # Use a composition of homographies to generate the final transform
        # that has to be applied
        transform_matrix = np.dot(np.dot(np.dot(rotation_matrix,
                                                translation_matrix),
                                         shear_matrix), zoom_matrix)
        h, w = x.shape[rows_idx], x.shape[cols_idx]
        transform_matrix = transform_matrix_offset_center(transform_matrix,
                                                          h, w)
        # Apply all the transformations together
        x = apply_transform(x, transform_matrix, fill_mode=fill_mode,
                            cval=cval, order=1, rows_idx=rows_idx,
                            cols_idx=cols_idx)
        if y is not None and len(y) > 0:
            y = apply_transform(y, transform_matrix, fill_mode=fill_mode,
                                cval=cvalMask, order=0, rows_idx=rows_idx,
                                cols_idx=cols_idx)

    # Horizontal flip
    if np.random.random() < horizontal_flip:  # 0 = disabled
        x = flip_axis(x, cols_idx)
        if y is not None and len(y) > 0:
            y = flip_axis(y, cols_idx)

    # Vertical flip
    if np.random.random() < vertical_flip:  # 0 = disabled
        x = flip_axis(x, rows_idx)
        if y is not None and len(y) > 0:
            y = flip_axis(y, rows_idx)

    # Spline warp
    if spline_warp:
        import SimpleITK as sitk
        warp_field = gen_warp_field(shape=(x.shape[rows_idx],
                                           x.shape[cols_idx]),
                                    sigma=warp_sigma,
                                    grid_size=warp_grid_size)
        x = apply_warp(x, warp_field,
                       interpolator=sitk.sitkLinear,
                       fill_mode=fill_mode,
                       fill_constant=cval,
                       rows_idx=rows_idx, cols_idx=cols_idx)
        if y is not None and len(y) > 0:
            y = np.round(apply_warp(y, warp_field,
                                    interpolator=sitk.sitkNearestNeighbor,
                                    fill_mode=fill_mode,
                                    fill_constant=cvalMask,
                                    rows_idx=rows_idx, cols_idx=cols_idx))

    # Crop
    # Expects axes with shape (..., 0, 1)
    # TODO: Add center crop
    if crop_size:
        # Reshape to (..., 0, 1)
        pattern = [el for el in range(x.ndim) if el != rows_idx and
                   el != cols_idx] + [rows_idx, cols_idx]
        inv_pattern = [pattern.index(el) for el in range(x.ndim)]
        x = x.transpose(pattern)

        crop = list(crop_size)
        pad = [0, 0]
        h, w = x.shape[-2:]

        # Compute amounts
        if crop[0] < h:
            # Do random crop
            top = np.random.randint(h - crop[0])
        else:
            # Set pad and reset crop
            pad[0] = crop[0] - h
            top, crop[0] = 0, h
        if crop[1] < w:
            # Do random crop
            left = np.random.randint(w - crop[1])
        else:
            # Set pad and reset crop
            pad[1] = crop[1] - w
            left, crop[1] = 0, w

        # Cropping
        x = x[..., top:top+crop[0], left:left+crop[1]]
        if y is not None and len(y) > 0:
            y = y.transpose(pattern)
            y = y[..., top:top+crop[0], left:left+crop[1]]
        # Padding
        if pad != [0, 0]:
            pad_pattern = ((0, 0),) * (x.ndim - 2) + (
                (pad[0]//2, pad[0] - pad[0]//2),
                (pad[1]//2, pad[1] - pad[1]//2))
            x = np.pad(x, pad_pattern, 'constant')
            y = np.pad(y, pad_pattern, 'constant', constant_values=void_label)

        x = x.transpose(inv_pattern)
        if y is not None and len(y) > 0:
            y = y.transpose(inv_pattern)

    if return_optical_flow:
        flow = optical_flow(x, rows_idx, cols_idx, chan_idx,
                            return_rgb=return_optical_flow=='rgb')
        x = np.concatenate((x, flow), axis=chan_idx)

    # Save augmented images
    if save_to_dir:
        import seaborn as sns
        fname = 'data_augm_{}.png'.format(np.random.randint(1e4))
        print ('Save to dir'.format(fname))
        cmap = sns.hls_palette(nclasses)
        save_img2(x, y, os.path.join(save_to_dir, fname),
                  cmap, void_label, rows_idx, cols_idx, chan_idx)

    # Undo extra dim
    if y is not None and len(y) > 0:
        y = y[..., 0]

    return x, y
