import numpy as np


# Make a random crop from image and mask of crop_size.
# Output size will be crop_size or smaller if the image
# size is smaller
def random_crop(img, mask, random_state,
                crop_size, patch_step=(20, 20),
                teacher_pred=None, teacher_soft=None):
    # Get image size
    image_size = img.shape

    # Find horizontal cropping indices
    if image_size[0] > crop_size[0]:
        bound1 = np.arange(0, image_size[0] - crop_size[0],
                           patch_step[0]).astype("int32")
        bound2 = np.arange(crop_size[0], image_size[0],
                           patch_step[0]).astype("int32")
        g1 = list(zip(bound1, bound2))
        random_state.shuffle(g1)
        lr = g1[0]
    else:
        lr = (0, image_size[0])

    # Find vertical cropping indices
    if image_size[1] > crop_size[1]:
        bound3 = np.arange(0, image_size[1] - crop_size[1],
                           patch_step[1]).astype("int32")
        bound4 = np.arange(crop_size[1], image_size[1],
                           patch_step[1]).astype("int32")
        g2 = list(zip(bound3, bound4))
        random_state.shuffle(g2)
        ud = g2[0]
    else:
        ud = (0, image_size[1])

    # Crop image and mask
    img = img[lr[0]:lr[1], ud[0]:ud[1]]
    mask = mask[lr[0]:lr[1], ud[0]:ud[1]]

    rval = [img, mask]

    if teacher_pred is not None:
        pred = teacher_pred[lr[0]:lr[1], ud[0]:ud[1]]
        rval = rval + [pred]
    if teacher_soft is not None:
        soft = teacher_soft[lr[0]:lr[1], ud[0]:ud[1]]
        rval = rval + [soft]

    return rval


def convert_01c_to_c01(image):
    image = np.swapaxes(image, 1, 2)
    image = np.swapaxes(image, 0, 1)

    return image


def convert_softmax_output(mask):
    mask = np.reshape(mask,
                      (mask.shape[0] *
                       mask.shape[1]))

    return mask
