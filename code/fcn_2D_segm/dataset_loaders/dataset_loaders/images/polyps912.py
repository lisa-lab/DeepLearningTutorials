import os
import time

import numpy as np

from dataset_loaders.parallel_loader import ThreadedDataset

floatX = 'float32'


class Polyps912Dataset(ThreadedDataset):
    '''The Endoluminal Scene Segmentation (EndoScene) of Colonoscopy Images
    benchmark

    The EndoScene dataset [1]_ consists of 912 frames extracted from 44
    colonoscopy sequences of 36 patients. The dataset combines both CVC-ColonDB
    and CVC-ClinicDB datasets of [2]_ and extends the dataset annotations to
    account for 4 different semantic classes.

    This loader is intended for the EndoScene dataset version containing 2
    semantic classes, namely polyp and background, plus a void class annotating
    the border of the images. However, it could be easily adapted to account
    for 3 or 4 classes.

    The dataset should be downloaded from [1]_ into the `shared_path`
    (that should be specified in the config.ini according to the
    instructions in ../README.md).

    Parameters
    ----------
    which_set: string
        A string in ['train', 'val', 'valid', 'test'], corresponding to
        the set to be returned.

     References
    ----------
    .. [1] http://adas.cvc.uab.es/endoscene/
    .. [2] https://endovis.grand-challenge.org/
    '''
    name = 'polyps912'
    non_void_nclasses = 2
    _void_labels = [2]

    # optional arguments
    data_shape = (384, 288, 3)
    _cmap = {
        0: (0, 0, 0),       # Background
        1: (255, 255, 255), # Polyp
        2: (128, 128, 128), # Void
        }
    _mask_labels = {0: 'Background', 1: 'Polyp', 2: 'Void'}

    _filenames = None

    @property
    def filenames(self):
        import glob

        if self._filenames is None:
            # Load filenames
            filenames = []

            # Get file names from images folder
            file_pattern = os.path.join(self.image_path, "*.bmp")
            file_names = glob.glob(file_pattern)
            # print (str(file_names))

            # Get raw filenames from file names list
            for file_name in file_names:
                path, file_name = os.path.split(file_name)
                file_name, ext = os.path.splitext(file_name)
                filenames.append(file_name)
                # print (file_name)

            # Save the filenames list
            self._filenames = filenames
        return self._filenames

    def __init__(self, which_set='train', *args, **kwargs):

        # Put which_set in canonical form: training, validation or testing
        if which_set in ("train", "training"):
            self.which_set = "train"
        elif which_set in ("val", "valid", "validation"):
            self.which_set = "valid"
        elif which_set in ("test", "testing"):
            self.which_set = "test"
        else:
            raise ValueError("Unknown set requested: %s" % which_set)

        # Define the images and mask paths
        self.image_path = os.path.join(self.path, self.which_set, 'images')
        self.mask_path = os.path.join(self.path, self.which_set, 'masks2')

        super(Polyps912Dataset, self).__init__(*args, **kwargs)

    def get_names(self):
        """Return a dict of names, per prefix/subset."""
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
        image_batch, mask_batch, filename_batch = [], [], []

        for prefix, img_name in sequence:

            img = io.imread(os.path.join(self.image_path, img_name + ".bmp"))
            img = img.astype(floatX) / 255.

            mask = np.array(io.imread(os.path.join(self.mask_path,
                                                   img_name + ".tif")))
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
    trainiter = Polyps912Dataset(
        which_set='train',
        batch_size=10,
        seq_per_subset=0,
        seq_length=0,
        data_augm_kwargs={
            'crop_size': (288, 384)},
        return_one_hot=True,
        return_01c=True,
        return_list=True,
        use_threads=False)

    validiter = Polyps912Dataset(
        which_set='valid',
        batch_size=1,
        seq_per_subset=0,
        seq_length=0,
        return_one_hot=True,
        return_01c=True,
        return_list=True,
        use_threads=False)

    testiter = Polyps912Dataset(
        which_set='test',
        batch_size=1,
        seq_per_subset=0,
        seq_length=0,
        return_one_hot=True,
        return_01c=True,
        return_list=True,
        use_threads=False)

    # Get number of classes
    nclasses = trainiter.nclasses
    print ("N classes: " + str(nclasses))
    void_labels = trainiter.void_labels
    print ("Void label: " + str(void_labels))

    # Training info
    train_nsamples = trainiter.nsamples
    train_batch_size = trainiter.batch_size
    train_nbatches = trainiter.nbatches
    print("Train n_images: {}, batch_size: {}, n_batches: {}".format(
        train_nsamples, train_batch_size, train_nbatches))

    # Validation info
    valid_nsamples = validiter.nsamples
    valid_batch_size = validiter.batch_size
    valid_nbatches = validiter.nbatches
    print("Validation n_images: {}, batch_size: {}, n_batches: {}".format(
        valid_nsamples, valid_batch_size, valid_nbatches))

    # Testing info
    test_nsamples = testiter.nsamples
    test_batch_size = testiter.batch_size
    test_nbatches = testiter.nbatches
    print("Test n_images: {}, batch_size: {}, n_batches: {}".format(
        test_nsamples, test_batch_size, test_nbatches))

    start = time.time()
    tot = 0
    max_epochs = 1

    for epoch in range(max_epochs):
        for mb in range(train_nbatches):
            train_group = trainiter.next()
            if train_group is None:
                raise RuntimeError('One batch was missing')

            # train_group checks
            assert train_group[0].ndim == 4
            assert train_group[0].shape[0] <= train_batch_size
            assert train_group[0].shape[1:] == (288, 384, 3)
            assert train_group[0].min() >= 0
            assert train_group[0].max() <= 1
            assert train_group[1].ndim == 4
            assert train_group[1].shape[0] <= train_batch_size
            assert train_group[1].shape[1:] == (288, 384, nclasses)

            # time.sleep approximates running some model
            time.sleep(1)
            stop = time.time()
            part = stop - start - 1
            start = stop
            tot += part
            print("Minibatch %s time: %s (%s)" % (str(mb), part, tot))


def run_tests():
    test()


if __name__ == '__main__':
    run_tests()
