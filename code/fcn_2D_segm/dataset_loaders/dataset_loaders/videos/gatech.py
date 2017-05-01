import os
import time

import numpy as np

from dataset_loaders.parallel_loader import ThreadedDataset
from dataset_loaders.utils_parallel_loader import natural_keys

floatX = 'float32'


class GatechDataset(ThreadedDataset):
    '''The Geometric Context from Video

    The Gatech dataset [1]_ consists of over 100 ground-truth annotated
    outdoor videos with over 20,000 frames.

    The dataset should be downloaded from [1]_ into the `shared_path`
    (that should be specified in the config.ini according to the
    instructions in ../README.md).

    Parameters
    ----------
    which_set: string
        A string in ['train', 'val', 'test'], corresponding to the set
        to be returned.
    split: float
        The percentage of the training data to be used for validation.
        The first `split`\% of the training set will be used for
        training and the rest for validation. Default: 0.75.

     References
    ----------
    .. [1] http://www.cc.gatech.edu/cpl/projects/videogeometriccontext/
    '''
    name = 'gatech'
    non_void_nclasses = 8
    _void_labels = [0]

    mean = [0.484375, 0.4987793, 0.46508789]
    std = [0.07699376, 0.06672145, 0.09592211]
    class_freqs = [0.32278032, 0.26515765, 0.21226492, 0.13986998, 0.01805662,
                   0.01745872, 0.01791921, 0.0041982,  0.00229439]
    # wtf, sky, ground, solid (buildings, etc), porous, cars, humans,
    # vert mix, main mix
    _cmap = {
        0: (255, 128, 0),      # wtf
        1: (255, 0, 0),        # sky (red)
        2: (0, 130, 180),      # ground (blue)
        3: (0, 255, 0),        # solid (buildings, etc) (green)
        4: (255, 255, 0),      # porous (yellow)
        5: (120, 0, 255),      # cars
        6: (255, 0, 255),      # humans (purple)
        7: (160, 160, 160),    # vert mix
        8: (64, 64, 64)}       # main mix
    _mask_labels = {0: 'wtf', 1: 'sky', 2: 'ground', 3: 'solid', 4: 'porous',
                    5: 'cars', 6: 'humans', 7: 'vert mix', 8: 'gen mix'}

    _filenames = None
    _prefix_list = None

    @property
    def prefix_list(self):
        if self._prefix_list is None:
            # Create a list of prefix out of the number of requested videos
            all_prefix_list = np.unique(np.array([el[:el.index('_')]
                                                  for el in self.filenames]))
            nvideos = len(all_prefix_list)
            nvideos_set = int(nvideos*self.split)
            self._prefix_list = all_prefix_list[nvideos_set:] \
                if "val" in self.which_set else all_prefix_list[:nvideos_set]

        return self._prefix_list

    @property
    def filenames(self):
        if self._filenames is None:
            # Get file names for this set
            self._filenames = os.listdir(self.image_path)
            self._filenames.sort(key=natural_keys)

            # Note: will get modified by prefix_list
        return self._filenames

    def __init__(self,
                 which_set='train',
                 split=.75,
                 *args, **kwargs):

        self.which_set = which_set

        # Prepare data paths
        if 'train' in self.which_set or 'val' in self.which_set:
            self.split = split
            if 'fcn8' in self.which_set:
                self.image_path = os.path.join(self.path, 'Images',
                                               'After_fcn8')
            else:
                self.image_path = os.path.join(self.path, 'Images',
                                               'Original')
            self.mask_path = os.path.join(self.path, 'Images', 'Ground_Truth')
        elif 'test' in self.which_set:
            self.image_path = os.path.join(self.path, 'Images_test',
                                           'Original')
            self.mask_path = os.path.join(self.path, 'Images_test',
                                          'Ground_Truth')
            self.split = split
            if 'fcn8' in self.which_set:
                raise RuntimeError('FCN8 outputs not available for test set')
        else:
            raise RuntimeError('Unknown set')

        super(GatechDataset, self).__init__(*args, **kwargs)

    def get_names(self):
        """Return a dict of names, per prefix/subset."""
        per_video_names = {}
        # Populate self.filenames and self.prefix_list
        filenames = self.filenames
        prefix_list = self.prefix_list

        # cycle through the different videos
        for prefix in prefix_list:
            per_video_names[prefix] = [el for el in filenames if
                                       el.startswith(prefix + '_')]
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
    trainiter = GatechDataset(
        which_set='train',
        batch_size=20,
        seq_per_subset=0,
        seq_length=0,
        data_augm_kwargs={
            'crop_size': (224, 224)},
        split=.75,
        return_one_hot=True,
        return_01c=True,
        return_list=True,
        use_threads=True)
    validiter = GatechDataset(
        which_set='valid',
        batch_size=1,
        seq_per_subset=0,
        seq_length=0,
        split=.75,
        return_one_hot=False,
        return_01c=True,
        return_list=True,
        use_threads=True)
    testiter = GatechDataset(
        which_set='test',
        batch_size=1,
        seq_per_subset=10,
        seq_length=10,
        split=1.,
        return_one_hot=False,
        return_01c=False,
        return_list=True,
        use_threads=True)

    train_nsamples = trainiter.nsamples
    valid_nsamples = validiter.nsamples
    test_nsamples = testiter.nsamples
    nclasses = testiter.nclasses
    nbatches = trainiter.nbatches
    train_batch_size = trainiter.batch_size
    valid_batch_size = validiter.batch_size
    test_batch_size = testiter.batch_size

    print("Train %d, valid %d, test %d" % (train_nsamples, valid_nsamples,
                                           test_nsamples))

    start = time.time()
    tot = 0
    max_epochs = 2

    for epoch in range(max_epochs):
        for mb in range(nbatches):
            train_group = trainiter.next()
            valid_group = validiter.next()
            test_group = testiter.next()
            if train_group is None or valid_group is None or \
               test_group is None:
                raise ValueError('.next() returned None!')

            # train_group checks
            assert train_group[0].ndim == 4
            assert train_group[0].shape[0] <= train_batch_size
            assert train_group[0].shape[1] == 224
            assert train_group[0].shape[2] == 224
            assert train_group[0].shape[3] == 3
            assert train_group[0].min() >= 0
            assert train_group[0].max() <= 1
            assert train_group[1].ndim == 4
            assert train_group[1].shape[0] <= train_batch_size
            assert train_group[1].shape[1] == 224
            assert train_group[1].shape[2] == 224
            assert train_group[1].shape[3] == nclasses

            # valid_group checks
            assert valid_group[0].ndim == 4
            assert valid_group[0].shape[0] <= valid_batch_size
            assert valid_group[0].shape[3] == 3
            assert valid_group[1].ndim == 3
            assert valid_group[1].shape[0] <= valid_batch_size
            assert valid_group[1].max() < nclasses

            # test_group checks
            assert test_group[0].ndim == 5
            assert test_group[0].shape[0] <= test_batch_size
            assert test_group[0].shape[1] == 10
            assert test_group[0].shape[2] == 3
            assert test_group[1].ndim == 4
            assert test_group[1].shape[0] <= test_batch_size
            assert test_group[1].shape[1] == 10
            assert test_group[1].max() < nclasses

            # time.sleep approximates running some model
            time.sleep(1)
            stop = time.time()
            part = stop - start - 1
            start = stop
            tot += part
            print("Minibatch %s - Threaded time: %s (%s)" % (str(mb), part,
                                                             tot))
    print("Test succesfull!!")


def test2():
    mean_time = {}
    for use_threads in [False, True]:
        trainiter = GatechDataset(
            which_set='train',
            batch_size=50,
            seq_per_subset=0,  # all of them
            seq_length=7,
            overlap=6,
            data_augm_kwargs={
                'crop_size': (224, 224)},
            split=.75,
            return_one_hot=True,
            return_01c=True,
            use_threads=use_threads,
            return_list=True,
            nthreads=5)

        train_nsamples = trainiter.nsamples
        nclasses = trainiter.nclasses
        nbatches = trainiter.nbatches
        train_batch_size = trainiter.batch_size

        print("Train %d" % (train_nsamples))

        start = time.time()
        tot = 0
        max_epochs = 1

        for epoch in range(max_epochs):
            for mb in range(nbatches):
                train_group = trainiter.next()

                # train_group checks
                assert train_group[0].ndim == 5
                assert train_group[0].shape[0] <= train_batch_size
                assert train_group[0].shape[1] == 7
                assert train_group[0].shape[2] == 224
                assert train_group[0].shape[3] == 224
                assert train_group[0].shape[4] == 3
                assert train_group[0].min() >= 0
                assert train_group[0].max() <= 1
                assert train_group[1].ndim == 5
                assert train_group[1].shape[0] <= train_batch_size
                assert train_group[1].shape[1] == 7
                assert train_group[1].shape[2] == 224
                assert train_group[1].shape[3] == 224
                assert train_group[1].shape[4] == nclasses

                # time.sleep approximates running some model
                time.sleep(1)
                stop = time.time()
                part = stop - start - 1
                start = stop
                tot += part
                print("Minibatch %s (Threaded %s): %s (%s)" %
                      (str(mb), str(use_threads), part, tot))
        mean_time[use_threads] = tot / nbatches*max_epochs
    print("Test succesfull!!")
    print("Mean times: %s (threaded) %s (unthreaded)" %
          (str(mean_time[True]), str(mean_time[False])))


def run_tests():
    test()
    test2()


if __name__ == '__main__':
    run_tests()
