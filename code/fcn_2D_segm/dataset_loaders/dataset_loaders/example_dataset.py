import numpy as np
import os

import dataset_loaders
from dataset_loaders.parallel_loader import ThreadedDataset


floatX = 'float32'


class ExampleDataset(ThreadedDataset):
    # Mandatory arguments
    # -------------------

    # A descriptive name for the dataset.
    name = 'example dataset'

    # The number of *non void* classes.
    non_void_nclasses = 10

    # Where the dataset will be saved *locally*.
    path = os.path.join(
        dataset_loaders.__path__[0], 'datasets', 'example_dataset')

    # The shared (typically on a network filesystem) path where the
    # dataset can be found to copy it locally at the first run.
    sharedpath = '/the/path/of/the/shared/location/of/the/data/'

    # A list of the ids of the void labels
    _void_labels = [128, 255]

    # Optional arguments
    # ------------------

    # The shape of the data *when constant*. Do not specify if images/videos
    # in the dataset have different shapes.
    data_shape = (360, 480, 3)

    # Set set_has_GT to False if the dataset has no ground truth. Else, set
    # it to False only for the sets that don't have GT. See `__init__`
    # for details.
    # set_has_GT = False

    # A list of classes labels. To be provided when the classes labels
    # (including the void ones) are not consecutive. If not provided,
    # the dataset loader will assume that the ids of the classes start
    # from 0 and are all consecutive.
    GTclasses = range(non_void_nclasses) + _void_labels

    # The dataset-wide statistics (either channel-wise or pixel-wise).
    # `extra/running_stats` contains utilities to compute them.
    mean = [0.1, 0.2, 0.3]
    std = [0.21, 0.22, 0.23]

    # The frequency of the classes of the dataset. Note that this field
    # should be a list of values for the *output* classes, i.e., the
    # classes after the mapping. In other terms, the frequency of the
    # void labels should be summed up together and provided as last
    # value of the array.
    # Note that:
    #   * the total length of the class_freqs list should be equal to
    #     `self.nclasses`;
    #   * `sum(self.class_freqs)` should be close to 1.
    class_freqs = [0.166666, 0.083333, 0.083333, 0.166666, 0.083333,
                   0.083333, 0.083333, 0.083333, 0.041666, 0.041666,
                   0.083333]

    # A *dictionary* of the form `class_id: (R, G, B)`. `class_id` is
    # the class id in the original data.
    _cmap = {
        0: (128, 128, 128),    # sky
        1: (128, 0, 0),        # building
        2: (192, 192, 128),    # column_pole
        3: (128, 64, 128),     # road
        4: (0, 0, 192),        # sidewalk
        5: (128, 128, 0),      # Tree
        6: (192, 128, 128),    # SignSymbol
        7: (64, 64, 128),      # Fence
        8: (64, 0, 128),       # Car
        9: (64, 64, 0),        # Pedestrian
        128: (0, 128, 192),    # Void
        255: (0, 0, 0)}        # Void

    # A *dictionary* of form `class_id: label`. `class_id` is the class
    # id in the original data.
    _mask_labels = {0: 'sky', 1: 'building', 2: 'column_pole', 3: 'road',
                    4: 'sidewalk', 5: 'tree', 6: 'sign', 7: 'fence', 8: 'car',
                    9: 'pedestrian', 128: 'void', 255: 'void'}

    def __init__(self, which_set='train', *args, **kwargs):

        self.which_set = 'val' if which_set == 'valid' else which_set
        if self.which_set == 'train':
            self.image_path = os.path.join(self.path, 'train', 'images')
            self.mask_path = os.path.join(self.path, 'train', 'GT')
        elif self.which_set == 'val':
            self.image_path = os.path.join(self.path, 'val', 'images')
            self.mask_path = os.path.join(self.path, 'val', 'GT')
        elif self.which_set == 'test':
            self.image_path = os.path.join(self.path, 'test', 'images')
            # Set set_has_GT to false if some set does not have the ground
            # truth
            self.set_has_GT = False
        else:
            raise RuntimeError('Unknown set: {}'.format(which_set))

        # Call the ThreadedDataset constructor. This will automatically
        # copy the dataset in self.path (the local path) if needed.
        super(ExampleDataset, self).__init__(*args, **kwargs)

    def get_names(self):
        """Return a dict of names, per subset."""
        names = {}
        # Get file names for this set
        for root, dirs, files in os.walk(self.image_path):
            # Take the last subdir as subset name
            subset = root[-root[::-1].index('/'):]
            names[subset] = files
        return names

    def load_sequence(self, sequence):
        """Load a sequence of images/frames

        Auxiliary function that loads a sequence of frames with
        the corresponding ground truth and their filenames.
        Returns a dict with the images in [0, 1], their corresponding
        labels, their subset (i.e. category, clip, prefix) and their
        filenames.
        """
        from skimage import io
        X = []
        Y = []
        F = []

        for prefix, frame in sequence:
            img = io.imread(os.path.join(self.image_path, frame))
            mask = io.imread(os.path.join(self.mask_path, frame))

            img = img.astype(floatX) / 255.
            mask = mask.astype('int32')

            X.append(img)
            Y.append(mask)
            F.append(frame)

        ret = {}
        ret['data'] = np.array(X)
        ret['labels'] = np.array(Y)
        ret['subset'] = prefix
        ret['filenames'] = np.array(F)
        return ret

    # NOTE:
    # Do not forget to add the dataset to the
    # `dataset_loaders/__init__.py` file.
