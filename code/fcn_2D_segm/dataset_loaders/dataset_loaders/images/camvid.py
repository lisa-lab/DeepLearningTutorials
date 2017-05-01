import numpy as np
import os
import time

from dataset_loaders.parallel_loader import ThreadedDataset


floatX = 'float32'


class CamvidDataset(ThreadedDataset):
    '''The CamVid motion based segmentation dataset

    The Cambridge-driving Labeled Video Database (CamVid) [1]_ provides
    high-quality videos acquired at 30 Hz with the corresponding
    semantically labeled masks at 1 Hz and in part, 15 Hz. The ground
    truth labels associate each pixel with one of 32 semantic classes.

    This loader is intended for the SegNet version of the CamVid dataset,
    that resizes the original data to 360 by 480 resolution and remaps
    the ground truth to a subset of 11 semantic classes, plus a void
    class.

    The dataset should be downloaded from [2]_ into the `shared_path`
    (that should be specified in the config.ini according to the
    instructions in ../README.md).

    Parameters
    ----------
    which_set: string
        A string in ['train', 'val', 'valid', 'test'], corresponding to
        the set to be returned.

     References
    ----------
    .. [1] http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/
    .. [2] https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid
    '''
    name = 'camvid'
    non_void_nclasses = 11
    _void_labels = [11]

    # optional arguments
    data_shape = (360, 480, 3)
    mean = [0.39068785, 0.40521392, 0.41434407]
    std = [0.29652068, 0.30514979, 0.30080369]

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
        10: (0, 128, 192),     # Bicyclist
        11: (0, 0, 0)}         # Void
    _mask_labels = {0: 'sky', 1: 'building', 2: 'column_pole', 3: 'road',
                    4: 'sidewalk', 5: 'tree', 6: 'sign', 7: 'fence', 8: 'car',
                    9: 'pedestrian', 10: 'byciclist', 11: 'void'}

    _filenames = None
    _prefix_list = None

    @property
    def prefix_list(self):
        if self._prefix_list is None:
            # Create a list of prefix out of the number of requested videos
            self._prefix_list = np.unique(np.array([el[:6]
                                                    for el in self.filenames]))

        return self._prefix_list

    @property
    def filenames(self):
        if self._filenames is None:
            # Get file names for this set and year
            filenames = []
            with open(os.path.join(self.path, self.which_set + '.txt')) as f:
                for fi in f.readlines():
                    raw_name = fi.strip()
                    raw_name = raw_name.split("/")[4]
                    raw_name = raw_name.strip()
                    filenames.append(raw_name)
            self._filenames = filenames
        return self._filenames

    def __init__(self, which_set='train', *args, **kwargs):

        self.which_set = "val" if which_set == "valid" else which_set
        if self.which_set == "train":
            self.image_path = os.path.join(self.path, "train")
            self.mask_path = os.path.join(self.path, "trainannot")
        elif self.which_set == "val":
            self.image_path = os.path.join(self.path, "val")
            self.mask_path = os.path.join(self.path, "valannot")
        elif self.which_set == "test":
            self.image_path = os.path.join(self.path, "test")
            self.mask_path = os.path.join(self.path, "testannot")
        elif self.which_set == 'trainval':
            self.image_path = os.path.join(self.path, "trainval")
            self.mask_path = os.path.join(self.path, "trainvalannot")

        # constructing the ThreadedDataset
        # it also creates/copies the dataset in self.path if not already there
        super(CamvidDataset, self).__init__(*args, **kwargs)

    def get_names(self):
        """Return a dict of names, per prefix/subset."""
        per_subset_names = {}
        # Populate self.filenames and self.prefix_list
        filenames = self.filenames
        prefix_list = self.prefix_list

        # cycle through the different videos
        for prefix in prefix_list:
            per_subset_names[prefix] = [el for el in filenames if
                                        el.startswith(prefix)]
        return per_subset_names

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


def test():
    trainiter = CamvidDataset(
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

    validiter = CamvidDataset(
        which_set='valid',
        batch_size=5,
        seq_per_subset=0,
        seq_length=0,
        data_augm_kwargs={
            'crop_size': (224, 224)},
        return_one_hot=True,
        return_01c=True,
        return_list=True,
        use_threads=False)

    train_nsamples = trainiter.nsamples
    nbatches = trainiter.nbatches
    print("Train %d" % (train_nsamples))

    valid_nsamples = validiter.nsamples
    print("Valid %d" % (valid_nsamples))

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
