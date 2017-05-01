import os
import time

import numpy as np
from PIL import Image
import shutil

import dataset_loaders
from dataset_loaders.parallel_loader import ThreadedDataset

floatX = 'float32'


class PascalVOCdataset(ThreadedDataset):
    name = 'pascal_voc'
    non_void_nclasses = 21
    _void_labels = [255]

    data_shape = (None, None, 3)
    mean = np.asarray([122.67891434, 116.66876762, 104.00698793]).astype(
        'float32')
    GTclasses = range(21) + [255]
    _cmap = {
        0: (0, 0, 0),           # background
        1: (255, 0, 0),         # aeroplane
        2: (192, 192, 128),     # bicycle
        3: (128, 64, 128),      # bird
        4: (0, 0, 255),         # boat
        5: (0, 255, 0),         # bottle
        6: (192, 128, 128),     # bus
        7: (64, 64, 128),       # car
        8: (64, 0, 128),        # cat
        9: (64, 64, 0),         # chair
        10: (0, 128, 192),      # cow
        11: (0, 255, 255),      # diningtable
        12: (255, 0, 255),      # dog
        13: (255, 128, 0),      # horse
        14: (0, 102, 102),      # motorbike
        15: (102, 0, 204),      # person
        16: (128, 255, 0),      # potted_plant
        17: (224, 224, 224),    # sheep
        18: (102, 0, 51),       # sofa
        19: (153, 76, 0),       # train
        20: (229, 244, 204),    # tv_monitor
        255: (255, 255, 255)    # void
    }
    _mask_labels = {0: 'background', 1: 'aeroplane', 2: 'bicycle', 3: 'bird',
                    4: 'boat', 5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat',
                    9: 'chair', 10: 'cow', 11: 'diningtable', 12: 'dog',
                    13: 'horse', 14: 'motorbike', 15: 'person',
                    16: 'potted_plant', 17: 'sheep', 18: 'sofa',
                    19: 'train', 20: 'tv_monitor', 255: 'void'}

    _filenames = None

    @property
    def filenames(self):
        # Get file names from txt file
        def get_file_names(file_name_txt, is_extra):
            is_extra = "_" if is_extra else ""
            filenames = {}
            with open(file_name_txt) as f:
                for fi in f.readlines():
                    raw_name = fi.strip()
                    prefix = raw_name.split('_')[0]
                    filenames.setdefault(prefix, []).append(is_extra +
                                                            raw_name)
            return filenames

        if self._filenames is None:
            # Load filenames
            if self.which_set == 'train_extra':
                filenames = get_file_names(self.txt_path_extra, True)
                file_txt = os.path.join(self.txt_path, "train.txt")
                for k, v in get_file_names(file_txt, False).iteritems():
                    filenames.setdefault(k, []).extend(v)
            else:
                file_txt = os.path.join(self.txt_path, self.which_set + ".txt")
                filenames = get_file_names(file_txt, False)

            self._filenames = filenames
        return self._filenames

    def __init__(self,
                 which_set="train",
                 year="VOC2012",
                 *args, **kwargs):

        self.which_set = "val" if which_set == "valid" else which_set
        if self.which_set not in ("train", "trainval", "train_extra", "val",
                                  "test"):
            raise ValueError("Unknown argument to which_set %s" %
                             self.which_set)
        if self.which_set == "test" and year != "VOC2012":
            raise ValueError("No test set for other than 2012 year")
        if self.which_set == 'test':
            self.set_has_GT = False

        self.year = year
        self.path_extra = os.path.join(
            dataset_loaders.__path__[0], 'datasets', 'PASCAL-VOC_Extra')
        self.sharedpath_extra = ('/data/lisa/exp/vazquezd/datasets/'
                                 'PASCAL_Extension/dataset/dataset10253/')

        self.txt_path = os.path.join(self.path, self.year,
                                     "ImageSets", "Segmentation")
        self.image_path = os.path.join(self.path, self.year, "JPEGImages")
        self.mask_path = os.path.join(self.path, self.year,
                                      "SegmentationClass")

        # Extra data
        self.txt_path_extra = os.path.join(self.path_extra,
                                           "train_nosegval.txt")
        self.image_path_extra = os.path.join(self.path_extra, "images")
        self.mask_path_extra = os.path.join(self.path_extra, "masks")

        # Copy the extra data to the local path if not existing
        if not os.path.exists(self.path_extra):
            print('The local path {} does not exist. Copying '
                  'dataset extra data...'.format(self.path_extra))
            shutil.copytree(self.sharedpath_extra, self.path_extra)
            print('Done.')

        super(PascalVOCdataset, self).__init__(*args, **kwargs)

    def get_names(self):
        """Return a dict of names, per prefix/subset."""
        return self.filenames

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

        # Load image
        for _, img_name in sequence:
            # Check if it is an image of the extra training dataset
            if img_name[0] == "_":
                image_path = self.image_path_extra
                mask_path = self.mask_path_extra
                img_name = img_name[1:]
            else:
                image_path = self.image_path
                mask_path = self.mask_path

            img = io.imread(os.path.join(image_path,
                                         img_name + ".jpg"))
            img = img.astype(floatX) / 255.

            # Load mask
            if self.which_set != "test":
                mask = np.array(Image.open(
                    os.path.join(mask_path, img_name + ".png")))
                mask = mask.astype('int32')

            # Add to minibatch
            image_batch.append(img)
            if self.which_set != "test":
                mask_batch.append(mask)
            filename_batch.append(img_name)

        ret = {}
        ret['data'] = np.array(image_batch)
        ret['labels'] = np.array(mask_batch)
        ret['subset'] = 'default'
        ret['filenames'] = np.array(filename_batch)
        return ret


def test():
    trainiter = PascalVOCdataset(
        which_set='train',
        batch_size=5,
        seq_per_subset=0,
        seq_length=0,
        data_augm_kwargs={
            'crop_size': (71, 71)},
        return_one_hot=True,
        return_01c=True,
        return_list=True,
        use_threads=False,
        nthreads=5)

    train_nsamples = trainiter.nsamples
    nclasses = trainiter.nclasses
    nbatches = trainiter.nbatches
    train_batch_size = trainiter.batch_size
    print("Train %d" % (train_nsamples))

    start = time.time()
    tot = 0
    max_epochs = 2

    for epoch in range(max_epochs):
        for mb in range(nbatches):
            train_group = trainiter.next()
            if train_group is None:
                raise RuntimeError('One batch was missing')

            # train_group checks
            assert train_group[0].ndim == 4
            assert train_group[0].shape[0] <= train_batch_size
            assert train_group[0].shape[1:] == (71, 71, 3)
            assert train_group[0].min() >= 0
            assert train_group[0].max() <= 1
            assert train_group[1].ndim == 4
            assert train_group[1].shape[0] <= train_batch_size
            assert train_group[1].shape[1:] == (71, 71, nclasses)

            # time.sleep approximates running some model
            time.sleep(1)
            stop = time.time()
            part = stop - start - 1
            start = stop
            tot += part
            print("Minibatch %s time: %s (%s)" % (str(mb), part, tot))


def test2():
    trainiter = PascalVOCdataset(
        which_set='train',
        batch_size=100,
        seq_per_subset=0,
        seq_length=0,
        data_augm_kwargs={
            'crop_size': (71, 71)},
        return_one_hot=True,
        return_01c=True,
        return_list=True,
        use_threads=False)

    trainiter_extra = PascalVOCdataset(
        which_set='train_extra',
        batch_size=100,
        seq_per_subset=0,
        seq_length=0,
        data_augm_kwargs={
            'crop_size': (71, 71)},
        return_one_hot=True,
        return_01c=True,
        return_list=True,
        use_threads=False)

    validiter = PascalVOCdataset(
        which_set='valid',
        batch_size=5,
        seq_per_subset=0,
        seq_length=0,
        data_augm_kwargs={
            'crop_size': (71, 71)},
        return_one_hot=True,
        return_01c=True,
        return_list=True,
        use_threads=False)

    testiter = PascalVOCdataset(
        which_set='test',
        batch_size=5,
        seq_per_subset=0,
        seq_length=0,
        data_augm_kwargs={
            'crop_size': (71, 71)},
        return_one_hot=True,
        return_01c=True,
        return_list=True,
        use_threads=False)

    train_nsamples = trainiter.nsamples
    nbatches = trainiter.nbatches
    print("Train %d" % (train_nsamples))

    train_nsamples_extra = trainiter_extra.nsamples
    valid_nsamples = validiter.nsamples
    test_nsamples = testiter.nsamples
    print("Train extra %d" % (train_nsamples_extra))
    print("Valid %d" % (valid_nsamples))
    print("Test %d" % (test_nsamples))

    # Simulate training
    max_epochs = 2
    start_training = time.time()
    for epoch in range(max_epochs):
        start_epoch = time.time()
        for mb in range(nbatches):
            start_batch = time.time()
            trainiter_extra.next()

            print("Minibatch {}: {}".format(mb, (time.time() - start_batch)))
        print("Epoch time: %s" % str(time.time() - start_epoch))
    print("Training time: %s" % str(time.time() - start_training))


def run_tests():
    test()
    test2()


if __name__ == '__main__':
    run_tests()
