import os
import time

import numpy as np
from PIL import Image

from dataset_loaders.parallel_loader import ThreadedDataset

floatX = 'float32'


class KITTIdataset(ThreadedDataset):
    '''The KITTI Vision Benchmark Suite for semantic segmentation

    KITTI Vision Benchmark is a dataset [1]_, which provides high-resolution
    videos acquired by equpipping a standard station wagon with two high-
    resolution color and grayscale video cameras. The dataset was captured by
    driving around Karlsruhe, in rural areas and on highways. Up to 15 cars and
    30 pedestrians are visible per image.

    This loader is intended for the semantic segmentation task of the KITTI
    dataset. Since KITTI does not provide a semantic segmentation benchmark,
    a number of researchers have annotated the images with semantic labels [2]_.

    This loader is intended for the KITTI segmentation benchmark in [3]_, which
    consists in 146 images and annotations from the original KITTI visual
    odometry dataset. The ground truth labels associate each pixel with one of
    11 semantic classes, plus one void class.

    The dataset should be downloaded from [3]_ into the `shared_path`
    (that should be specified in the config.ini according to the
    instructions in ../README.md).

    Parameters
    ----------
    which_set: string
        A string in ['train', 'val', 'valid', 'test'], corresponding to
        the set to be returned.
    split: float
        A float indicating the dataset split between training and validation.
        For example, if split=0.85, 85\% of the images will be used for training,
        whereas 15\% will be used for validation.

     References
    ----------
    .. [1] http://www.cvlibs.net/datasets/kitti/index.php
    .. [2] http://www.cvlibs.net/datasets/kitti/eval_semantics.php
    .. [3] http://adas.cvc.uab.es/s2uad/
    '''
    name = 'kitti'
    non_void_nclasses = 11
    _void_labels = [11]

    # optional arguments
    # mean = np.asarray([122.67891434, 116.66876762, 104.00698793]).astype(
    #    'float32')
    mean = [0.35675976, 0.37380189, 0.3764753]
    std = [0.32064945, 0.32098866, 0.32325324]
    _cmap = {
        0: (128, 128, 128),    # Sky
        1: (128, 0, 0),        # Building
        2: (128, 64, 128),     # Road
        3: (0, 0, 192),        # Sidewalk
        4: (64, 64, 128),      # Fence
        5: (128, 128, 0),      # Vegetation
        6: (192, 192, 128),    # Pole
        7: (64, 0, 128),       # Car
        8: (192, 128, 128),    # Sign
        9: (64, 64, 0),        # Pedestrian
        10: (0, 128, 192),     # Cyclist
        11: (0, 0, 0)          # Void
    }
    _mask_labels = {0: 'Sky', 1: 'Building', 2: 'Road', 3: 'Sidewalk',
                    4: 'Fence', 5: 'Vegetation', 6: 'Pole', 7: 'Car',
                    8: 'Sign', 9: 'Pedestrian', 10: 'Cyclist', 11: 'void'}

    _filenames = None

    @property
    def filenames(self):
        import glob

        if self._filenames is None:
            # Load filenames
            filenames = []

            # Get file names from images folder
            file_pattern = os.path.join(self.image_path, "*.png")
            file_names = glob.glob(file_pattern)

            # Get raw filenames from file names list
            for file_name in file_names:
                path, file_name = os.path.split(file_name)
                file_name, ext = os.path.splitext(file_name)
                filenames.append(file_name)

            nfiles = len(filenames)
            if self.which_set == 'train':
                filenames = filenames[:int(nfiles*self.split)]
            elif self.which_set == 'val':
                filenames = filenames[-(nfiles - int(nfiles*self.split)):]

            # Save the filenames list
            self._filenames = filenames

        return self._filenames

    def __init__(self,
                 which_set="train",
                 split=0.85,
                 *args, **kwargs):

        self.which_set = "val" if which_set == "valid" else which_set
        if self.which_set not in ("train", "val", 'test', 'trainval'):
            raise ValueError("Unknown argument to which_set %s" %
                             self.which_set)

        if self.which_set == 'train':
            set_folder = 'Training_00/'
            self.split = split
        elif self.which_set == 'val':
            set_folder = 'Training_00/'
            self.split = split
        elif self.which_set == 'test':
            set_folder = 'Validation_07/'
            self.split = 1.0
        else:
            raise ValueError

        self.image_path = os.path.join(self.path, set_folder, "RGB")
        self.mask_path = os.path.join(self.path, set_folder, "GT_ind")

        super(KITTIdataset, self).__init__(*args, **kwargs)

    def get_names(self):
        """Return a dict of names, per prefix/subset."""

        # TODO: does kitty have prefixes/categories?
        return {'default': self.filenames}

    def load_sequence(self, sequence):
        """Load a sequence of images/frames

        Auxiliary function that loads a sequence of frames with
        the corresponding ground truth and their filenames.
        Returns a dict with the images in [0, 1], their corresponding
        labels, their subset (i.e. category, clip, prefix) and their
        filenames.
        """
        from skimage import io
        image_batch = []
        mask_batch = []
        filename_batch = []

        for prefix, img_name in sequence:
            # Load image
            img = io.imread(os.path.join(self.image_path, img_name + ".png"))
            img = img.astype(floatX) / 255.

            # Load mask
            mask = np.array(Image.open(
                    os.path.join(self.mask_path, img_name + ".png")))
            mask = mask.astype('int32')

            # Add to minibatch
            image_batch.append(img)
            mask_batch.append(mask)
            filename_batch.append(img_name)

        ret = {}
        ret['data'] = np.array(image_batch)
        ret['labels'] = np.array(mask_batch)
        ret['subset'] = prefix
        ret['filenames'] = np.array(filename_batch)
        return ret


def test():
    trainiter = KITTIdataset(
        which_set='train',
        batch_size=10,
        seq_per_subset=0,
        seq_length=0,
        data_augm_kwargs={
            'crop_size': (224, 224)},
        return_one_hot=True,
        return_01c=True,
        return_list=True,
        use_threads=True)

    validiter = KITTIdataset(
        which_set='valid',
        batch_size=1,
        seq_per_subset=0,
        seq_length=0,
        data_augm_kwargs={},
        return_one_hot=True,
        return_01c=True,
        return_list=True,
        use_threads=False)

    testiter = KITTIdataset(
        which_set='test',
        batch_size=1,
        seq_per_subset=0,
        seq_length=0,
        data_augm_kwargs={},
        return_one_hot=True,
        return_01c=True,
        return_list=True,
        use_threads=False)

    train_nsamples = trainiter.nsamples
    nbatches = trainiter.nbatches
    print("Train %d" % (train_nsamples))

    valid_nsamples = validiter.nsamples
    print("Valid %d" % (valid_nsamples))

    test_nsamples = testiter.nsamples
    print("Test %d" % (test_nsamples))

    # Simulate training
    max_epochs = 2
    start_training = time.time()
    for epoch in range(max_epochs):
        start_epoch = time.time()
        for mb in range(nbatches):
            start_batch = time.time()
            trainiter.next()
            print("Minibatch {}: {} seg".format(mb, (time.time() -
                                                     start_batch)))
        print("Epoch time: %s" % str(time.time() - start_epoch))
    print("Training time: %s" % str(time.time() - start_training))


def run_tests():
    test()


if __name__ == '__main__':
    run_tests()
