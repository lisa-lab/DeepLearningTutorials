import os
import time

import numpy as np

from dataset_loaders.parallel_loader import ThreadedDataset
from dataset_loaders.utils_parallel_loader import natural_keys

floatX = 'float32'


class DavisDataset(ThreadedDataset):
    name = 'davis'
    non_void_nclasses = 2
    _void_labels = []

    # NOTE: we only load the 480p
    # 1080p images are either (1920, 1080) or (1600, 900)
    data_shape = (854, 480, 3)
    _cmap = {
        0: (255, 255, 255),        # background
        1: (0, 0, 0)}              # foreground
    _mask_labels = {0: 'background', 1: 'foreground'}

    _filenames = None
    _prefix_list = None

    @property
    def prefix_list(self):
        if self._prefix_list is None:
            # Create a list of prefix out of the number of requested videos
            all_prefix_list = np.unique(np.array(os.listdir(self.image_path)))
            nvideos = len(all_prefix_list)
            nvideos_set = int(nvideos*self.split)
            self._prefix_list = all_prefix_list[nvideos_set:] \
                if "val" in self.which_set else all_prefix_list[:nvideos_set]

        return self._prefix_list

    @property
    def filenames(self):
        if self._filenames is None:
            self._filenames = []
            # Get file names for this set
            for root, dirs, files in os.walk(self.image_path):
                for name in files:
                        self._filenames.append(os.path.join(
                          root[-root[::-1].index('/'):], name[:-3]))

            self._filenames.sort(key=natural_keys)

            # Note: will get modified by prefix_list
        return self._filenames

    def __init__(self,
                 which_set='train',
                 threshold_masks=False,
                 split=.75,
                 *args, **kwargs):

        self.which_set = which_set
        self.threshold_masks = threshold_masks

        # Prepare data paths
        if 'train' in self.which_set or 'val' in self.which_set:
            self.split = split
            self.image_path = os.path.join(self.path,
                                           'JPEGImages', '480p',
                                           'training')
            self.mask_path = os.path.join(self.path,
                                          'Annotations', '480p',
                                          'training')
        elif 'test' in self.which_set:
            self.image_path = os.path.join(self.path,
                                           'JPEGImages', '480p', 'test')
            self.mask_path = os.path.join(self.path,
                                          'Annotations', '480p', 'test')
            self.split = 1.
        else:
            raise RuntimeError('Unknown set')

        super(DavisDataset, self).__init__(*args, **kwargs)

    def get_names(self):
        """Return a dict of names, per prefix/subset."""
        per_video_names = {}

        # Populate self.filenames and self.prefix_list
        filenames = self.filenames
        prefix_list = self.prefix_list

        # cycle through the different videos
        for prefix in prefix_list:
            exp_prefix = prefix + '/'
            per_video_names[prefix] = [el.lstrip(exp_prefix) for el in
                                       filenames if el.startswith(exp_prefix)]
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

        for prefix, frame_name in sequence:
            frame = prefix + '/' + frame_name

            img = io.imread(os.path.join(self.image_path, frame + 'jpg'))
            mask = io.imread(os.path.join(self.mask_path, frame + 'png'))

            img = img.astype(floatX) / 255.
            mask = (mask / 255).astype('int32')

            X.append(img)
            Y.append(mask)
            F.append(frame_name + 'jpg')

        ret = {}
        ret['data'] = np.array(X)
        ret['labels'] = np.array(Y)
        ret['subset'] = prefix
        ret['filenames'] = np.array(F)
        return ret


def test():
    trainiter = DavisDataset(
        which_set='train',
        batch_size=20,
        seq_per_subset=0,
        seq_length=0,
        overlap=0,
        data_augm_kwargs={
            'crop_size': (224, 224)},
        split=0.75,
        return_one_hot=True,
        return_01c=True,
        use_threads=True,
        shuffle_at_each_epoch=False)
    validiter = DavisDataset(
        which_set='valid',
        batch_size=1,
        seq_per_subset=0,
        seq_length=0,
        overlap=0,
        split=.75,
        return_one_hot=False,
        return_01c=True,
        use_threads=True,
        shuffle_at_each_epoch=False)
    testiter = DavisDataset(
        which_set='test',
        batch_size=1,
        seq_per_subset=0,
        seq_length=0,
        overlap=0,
        split=1.,
        return_one_hot=False,
        return_01c=False,
        use_threads=True)

    train_nsamples = trainiter.nsamples
    valid_nsamples = validiter.nsamples
    test_nsamples = testiter.nsamples
    nbatches = trainiter.nbatches

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

            # time.sleep approximates running some model
            time.sleep(1)
            stop = time.time()
            part = stop - start - 1
            start = stop
            tot += part
            print("Minibatch %s - Threaded time: %s (%s)" % (str(mb), part,
                                                             tot))


def run_tests():
    test()


if __name__ == '__main__':
    run_tests()
