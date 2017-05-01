import numpy as np
import os
import time

from dataset_loaders.parallel_loader import ThreadedDataset
from dataset_loaders.utils_parallel_loader import natural_keys


floatX = 'float32'


class CityscapesDataset(ThreadedDataset):
    '''The cityscapes dataset

    To prepare the dataset with the correct class mapping, use the scripts
    provided by the authors: [2]_

    Notes
    -----
    To change the class mapping it suffices to edit
    `cityscapesscripts/helpers/labels.py`:
        * id = -1  ignores the class when building the GT
        * id = 255 considers the class as void/unlabeled when building the GT
    and run the script
    `cityscapesscripts/preparation/createTrainIdLabelImgs.py`
    in order to generate the new ground truth images. The `_cmap`,
    `_mask_labels`, `GTclasses`, `_void_labels` and `non_void_nclasses`
    attributes of the dataset class must be modified accordingly.

    The dataset should be downloaded from [1]_ into the `shared_path`
    (that should be specified in the config.ini according to the
    instructions in ../README.md).

    Notes
    -----
    To submit the results to the evaluation server the classes have to be
    remapped to the original 0 to 33 range.

    Parameters
    ----------
    which_set: string
        A string in ['train', 'val', 'valid', 'test'], corresponding to
        the set to be returned.

    References
    ----------
    .. [1] https://www.cityscapes-dataset.com
    .. [2] https://github.com/mcordts/cityscapesScripts
    '''
    name = 'cityscapes'
    non_void_nclasses = 19
    _void_labels = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
    GTclasses = range(34)
    GTclasses = GTclasses + [-1]

    # optional arguments
    data_shape = (2048, 1024, 3)
    _cmap = {
        0: (0, 0, 0),           # unlabeled
        1: (0, 0, 0),           # ego vehicle
        2: (0, 0, 0),           # rectification border
        3: (0, 0, 0),           # out of roi
        4: (0, 0, 0),           # static
        5: (111, 74, 0),        # dynamic
        6: (81,  0, 81),        # ground
        7: (128, 64, 128),      # road
        8: (244, 35, 232),      # sidewalk
        9: (250, 170, 160),     # parking
        10: (230, 150, 140),    # rail track
        11: (70, 70, 70),       # building
        12: (102, 102, 156),    # wall
        13: (190, 153, 153),    # fence
        14: (180, 165, 180),    # guard rail
        15: (150, 100, 100),    # bridge
        16: (150, 120, 90),     # tunnel
        17: (153, 153, 153),    # pole
        18: (153, 153, 153),    # polegroup
        19: (250, 170, 30),     # traffic light
        20: (220, 220,  0),     # traffic sign
        21: (107, 142, 35),     # vegetation
        22: (152, 251, 152),    # terrain
        23: (0, 130, 180),      # sky
        24: (220, 20, 60),      # person
        25: (255, 0, 0),        # rider
        26: (0, 0, 142),        # car
        27: (0, 0, 70),         # truck
        28: (0, 60, 100),       # bus
        29: (0,  0, 90),        # caravan
        30: (0,  0, 110),       # trailer
        31: (0, 80, 100),       # train
        32: (0, 0, 230),        # motorcycle
        33: (119, 11, 32),      # bicycle
        -1: (0, 0, 142)         # license plate
        }

    _mask_labels = {
        0: 'unlabeled',
        1: 'ego vehicle',
        2: 'rectification border',
        3: 'out of roi',
        4: 'static',
        5: 'dynamic',
        6: 'ground',
        7: 'road',
        8: 'sidewalk',
        9: 'parking',
        10: 'rail track',
        11: 'building',
        12: 'wall',
        13: 'fence',
        14: 'guard rail',
        15: 'bridge',
        16: 'tunnel',
        17: 'pole',
        18: 'polegroup',
        19: 'traffic light',
        20: 'traffic sign',
        21: 'vegetation',
        22: 'terrain',
        23: 'sky',
        24: 'person',
        25: 'rider',
        26: 'car',
        27: 'truck',
        28: 'bus',
        29: 'caravan',
        30: 'trailer',
        31: 'train',
        32: 'motorcycle',
        33: 'bicycle',
        -1: 'license plate'
    }

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
            self._filenames = []
            # Get file names for this set
            for root, dirs, files in os.walk(self.image_path):
                for name in files:
                        self._filenames.append(os.path.join(
                          root[-root[::-1].index('/'):], name))

            self._filenames.sort(key=natural_keys)

            # Note: will get modified by prefix_list
        return self._filenames

    def __init__(self, which_set='train', *args, **kwargs):

        self.which_set = "val" if which_set == "valid" else which_set
        if self.which_set not in ['train', 'val', 'test']:
            raise NotImplementedError('Unknown set: ' + which_set)
        if self.which_set == 'test':
            self.set_has_GT = False

        self.image_path = os.path.join(self.path,
                                       "leftImg8bit",
                                       self.which_set)
        self.mask_path = os.path.join(self.path,
                                      "gtFine",
                                      self.which_set)

        # This also creates/copies the dataset in self.path if missing
        super(CityscapesDataset, self).__init__(*args, **kwargs)

    def get_names(self):
        """Return a dict of names, per prefix/subset."""
        per_video_names = {}
        # Populate self.filenames and self.prefix_list
        prefix_list = self.prefix_list

        # cycle through the different videos
        for prefix in prefix_list:
            per_video_names[prefix] = [el for el in self.filenames
                                       if prefix in el]
        return per_video_names

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
            img = img.astype(floatX) / 255.
            X.append(img)
            F.append(frame)

            if self.set_has_GT:
                mask_filename = frame.replace("leftImg8bit",
                                              "gtFine_labelIds")
                mask = io.imread(os.path.join(self.mask_path, mask_filename))
                mask = mask.astype('int32')
                Y.append(mask)

        ret = {}
        ret['data'] = np.array(X)
        ret['labels'] = np.array(Y)
        ret['subset'] = prefix
        ret['filenames'] = np.array(F)
        return ret


def test1():
    d = CityscapesDataset(
        which_set='train',
        batch_size=5,
        seq_per_subset=4,
        seq_length=0,
        data_augm_kwargs={
            'crop_size': (224, 224)})
    start = time.time()
    n_minibatches_to_run = 1000
    tot = 0

    for mb in range(n_minibatches_to_run):
        image_group = d.next()
        if image_group is None:
            raise NotImplementedError()

        # time.sleep approximates running some model
        time.sleep(1)
        stop = time.time()
        part = stop - start - 1
        start = stop
        tot += part
        print("Minibatch %s time: %s (%s)" % (str(mb), part, tot))
    print("TEST 1 PASSED!!\n\n")


def test2():
    d = CityscapesDataset(
        which_set='train',
        batch_size=5,
        seq_per_subset=0,
        seq_length=10,
        overlap=9,
        return_one_hot=True,
        return_list=True,
        data_augm_kwargs={
            'crop_size': (224, 224)},
        use_threads=True)

    start = time.time()
    tot = 0
    for i, _ in enumerate(range(d.nbatches)):
        image_group = d.next()
        if image_group is None:
            raise RuntimeError()
        sh = image_group[0].shape
        assert(sh[0] <= 5)
        assert(sh[1] == 10)
        # time.sleep approximates running some model
        time.sleep(1)
        stop = time.time()
        part = stop - start - 1
        start = stop
        tot += part
        print("Minibatch %i/%i time: %s (%s)" % (i, d.nbatches, part, tot))
    print("TEST 2 PASSED!! \n\n")


def test3():
    trainiter = CityscapesDataset(
        which_set='val',
        batch_size=5,
        seq_per_subset=0,
        seq_length=0,
        data_augm_kwargs={
            'crop_size': (224, 224)},
        return_one_hot=False,
        return_01c=True,
        use_threads=True,
        return_list=True,
        nthreads=8)
    train_nsamples = trainiter.nsamples
    nclasses = trainiter.nclasses
    nbatches = 500
    train_batch_size = trainiter.batch_size
    print("Train %d" % (train_nsamples))
    trainiter.cmap
    max_epochs = 5

    for epoch in range(max_epochs):
        for mb in range(nbatches):
            start = time.time()
            train_group = trainiter.next()

            # train_group checks
            assert train_group[0].ndim == 4
            assert train_group[0].shape[0] <= train_batch_size
            assert train_group[0].shape[1] == 224
            assert train_group[0].shape[2] == 224
            assert train_group[0].shape[3] == 3
            assert train_group[0].min() >= 0
            assert train_group[0].max() <= 1

            if trainiter.set_has_GT:
                assert train_group[1].shape[0] <= train_batch_size
                assert train_group[1].shape[1] == 224
                assert train_group[1].shape[2] == 224

                if trainiter.return_one_hot:
                    assert train_group[1].ndim == 4
                    assert train_group[1].shape[3] == nclasses
                else:
                    assert train_group[1].ndim == 3

            # time.sleep approximates running some model
            time.sleep(0.1)
            stop = time.time()
            tot = stop - start
            print("Threaded time: %s" % (tot))
            print("Epoch %i, minibatch %i/%i" % (epoch, mb, nbatches))
        print('End of epoch --> should reset!')
        time.sleep(2)


def run_tests():
    test1()
    test2()
    test3()


if __name__ == '__main__':
    run_tests()
