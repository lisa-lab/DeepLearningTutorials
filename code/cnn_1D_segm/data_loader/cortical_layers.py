import os
import time

import numpy as np
from PIL import Image
import re
import warnings

from dataset_loaders.parallel_loader import ThreadedDataset
from parallel_loader_1D import ThreadedDataset_1D

floatX = 'float32'

class Cortical4LayersDataset(ThreadedDataset_1D):
    '''The Cortical Layers Dataset.
    Parameters
    ----------
    which_set: string
        A string in ['train', 'val', 'valid', 'test'], corresponding to
        the set to be returned.
    split: float
        A float indicating the dataset split between training and validation.
        For example, if split=0.85, 85\% of the images will be used for training,
        whereas 15\% will be used for validation.
    '''
    name = 'cortical_layers'

    non_void_nclasses = 6
    GTclasses = [0, 1, 3, 4, 6, 7]
    _cmap = {
        0: (128, 128, 128),    # padding
        1: (128, 0, 0),        # layers 1-2
        3: (128, 64, 128),     # layer 3
        4: (0, 0, 128),        # layers 4-5
        6: (64, 64, 128),      # layer 6
        7: (128, 128, 0),      # non-sense
    }
    _mask_labels = {0: 'padding', 1: 'layers12', 3: 'layer3', 4: 'layer45',
                    6: 'layer6', 7: 'nonsense'}
    _void_labels = []


    _filenames = None

    @property
    def filenames(self):

        if self._filenames is None:
            # Load filenames
            nfiles = sum(1 for line in open(self.mask_path))
            filenames = range(nfiles)
            np.random.seed(1609)
            np.random.shuffle(filenames)

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
                 shuffle_at_each_epoch = True,
                 smooth_or_raw = 'raw',
                 *args, **kwargs):
        self.task = 'segmentation'
        self.n_layers = 4
        n_layers_path = str(self.n_layers)+"layers_segmentation"

        self.which_set = "val" if which_set == "valid" else which_set
        if self.which_set not in ("train", "val", 'test'):
            raise ValueError("Unknown argument to which_set %s" %
                             self.which_set)

        self.split = split

        self.image_path_raw =  os.path.join(self.path,n_layers_path,"training_raw.txt")
        self.image_path_smooth =  os.path.join(self.path,n_layers_path, "training_geo.txt")
        self.mask_path = os.path.join(self.path,n_layers_path, "training_cls.txt")
        self.smooth_raw_both = smooth_or_raw
        #
        # print 'raw path', self.image_path_raw
        # print 'smooth path', self.image_path_smooth
        # print 'cls path', self.mask_path
        # print 'smooth or raw', self.smooth_raw_both

        if smooth_or_raw == 'both':
            self.data_shape = (200,2)
        else :
            self.data_shape = (200,1)

        super(Cortical4LayersDataset, self).__init__(*args, **kwargs)

    def get_names(self):
        """Return a dict of names, per prefix/subset."""

        return {'default': self.filenames}


class Cortical6LayersDataset(ThreadedDataset_1D):
    '''The Cortical Layers Dataset.
    Parameters
    ----------
    which_set: string
        A string in ['train', 'val', 'valid', 'test'], corresponding to
        the set to be returned.
    split: float
        A float indicating the dataset split between training and validation.
        For example, if split=0.85, 85\% of the images will be used for training,
        whereas 15\% will be used for validation.
    '''
    name = 'cortical_layers'

    non_void_nclasses = 7
    GTclasses = [0, 1, 2, 3, 4, 5, 6]
    _cmap = {
        0: (128, 128, 128),    # padding
        1: (128, 0, 0),        # layer 1
        2: (128, 64, ),        # layer 2
        3: (128, 64, 128),     # layer 3
        4: (0, 0, 128),        # layer 4
        5: (0, 0, 64),        # layer 5
        6: (64, 64, 128),      # layer 6
        #7: (128, 128, 0),      # non-sense
    }
    _mask_labels = {0: 'padding', 1: 'layers1', 2: 'layer2', 3: 'layer3',
                    4: 'layer4', 5: 'layer5',   6: 'layer6'}
    _void_labels = []


    _filenames = None

    @property
    def filenames(self):

        if self._filenames is None:
            # Load filenames
            nfiles = sum(1 for line in open(self.mask_path))
            filenames = range(nfiles)
            np.random.seed(1609)
            np.random.shuffle(filenames)

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
                 shuffle_at_each_epoch = True,
                 smooth_or_raw = 'raw',
                 *args, **kwargs):

        self.task = 'segmentation'

        self.n_layers = 6
        n_layers_path = str(self.n_layers)+"layers_segmentation"

        self.which_set = "val" if which_set == "valid" else which_set
        if self.which_set not in ("train", "val", 'test'):
            raise ValueError("Unknown argument to which_set %s" %
                             self.which_set)

        self.split = split

        self.image_path_raw =  os.path.join(self.path,n_layers_path,"training_raw.txt")
        self.image_path_smooth =  os.path.join(self.path,n_layers_path, "training_geo.txt")
        self.mask_path = os.path.join(self.path,n_layers_path, "training_cls.txt")
        self.regions_path = os.path.join(self.path, n_layers_path, "training_regions.txt")

        self.smooth_raw_both = smooth_or_raw

        # print 'raw path', self.image_path_raw
        # print 'smooth path', self.image_path_smooth
        # print 'cls path', self.mask_path
        # print 'smooth or raw', self.smooth_raw_both

        if smooth_or_raw == 'both':
            self.data_shape = (200,2)
        else :
            self.data_shape = (200,1)

        super(Cortical6LayersDataset, self).__init__(*args, **kwargs)

    def get_names(self):
        """Return a dict of names, per prefix/subset."""

        return {'default': self.filenames}


class ParcellationDataset(ThreadedDataset_1D):
    '''The Cortical Layers Dataset.
    Parameters
    ----------
    which_set: string
        A string in ['train', 'val', 'valid', 'test'], corresponding to
        the set to be returned.
    split: float
        A float indicating the dataset split between training and validation.
        For example, if split=0.85, 85\% of the images will be used for training,
        whereas 15\% will be used for validation.
    '''
    name = 'cortical_layers'

    non_void_nclasses = 8 #regions 1 to 8
    GTclasses = [1, 2, 3, 4, 5, 6, 7, 8] #??? use 0-7 or 1-8?


    _cmap = {
        1: (128, 0, 0),         # region 1
        2: (128, 64, ),         # region 2
        3: (128, 64, 128),      # region 3
        4: (0, 0, 128),         # region 4
        5: (0, 0, 64),          # region 5
        6: (64, 64, 128),       # region 6
        7: (128, 128, 0),       # region 7
        8 : (0,0,0),            # region 8
    }
    _mask_labels = {1: 'region1', 2: 'region1', 3: 'region1', 4: 'region1',
                    5: 'region1', 6: 'region1', 7: 'region1', 8: 'region1', }
    _void_labels = []


    _filenames = None

    @property
    def filenames(self):

        if self._filenames is None:
            # Load filenames
            nfiles = sum(1 for line in open(self.mask_path))
            print self.which_set, self.mask_path, nfiles
            filenames = range(nfiles)
            np.random.seed(1609)
            np.random.shuffle(filenames)


            # Save the filenames list
            self._filenames = filenames

        return self._filenames

    def __init__(self,
                 which_set="train",
                 split=0.85,
                 shuffle_at_each_epoch = True,
                 smooth_or_raw = 'both',
                 *args, **kwargs):

        folder = 'Parcellation'

        self.task = 'classification'

        self.which_set = "val" if which_set == "valid" else which_set
        if self.which_set not in ("train", "val", 'test'):
            raise ValueError("Unknown argument to which_set %s" %
                             self.which_set)

        self.split = split

        self.train_image_path_raw = os.path.join(self.path,folder,"training_raw.txt")
        self.train_image_path_smooth = os.path.join(self.path,folder,"training_raw.txt")
        self.train_labels = os.path.join(self.path, folder, "training_cls.txt")

        self.test_image_path_raw = os.path.join(self.path,folder,"testing_raw.txt")
        self.test_image_path_smooth = os.path.join(self.path,folder,"testing_smooth.txt")
        self.test_labels = os.path.join(self.path, folder, "testing_cls.txt")

        if self.which_set=='train':
            self.image_path_raw = self.train_image_path_raw
            self.image_path_smooth = self.train_image_path_smooth
            self.mask_path = self.train_labels

        elif self.which_set=='val':
            self.image_path_raw = self.test_image_path_raw
            self.image_path_smooth = self.test_image_path_smooth
            self.mask_path = self.test_labels



        self.smooth_raw_both = smooth_or_raw

        # print 'raw path', self.image_path_raw
        # print 'smooth path', self.image_path_smooth
        # print 'cls path', self.mask_path
        # print 'smooth or raw', self.smooth_raw_both

        if smooth_or_raw == 'both':
            self.data_shape = (200,2)
        else :
            self.data_shape = (200,1)

        super(ParcellationDataset, self).__init__(*args, **kwargs)

    def get_names(self):
        """Return a dict of names, per prefix/subset."""

        return {'default': self.filenames}

def test_6layers():
    trainiter = Cortical6LayersDataset(
        which_set='train',
        smooth_or_raw = 'smooth',
        batch_size=5,
        data_augm_kwargs={},
        return_one_hot=False,
        return_01c=False,
        return_list=True,
        use_threads=False)

    validiter = Cortical6LayersDataset(
        which_set='valid',
        smooth_or_raw = 'smooth',
        batch_size=5,
        data_augm_kwargs={},
        return_one_hot=False,
        return_01c=False,
        return_list=True,
        use_threads=False)

    validiter2 = Cortical6LayersDataset(
        which_set='valid',
        smooth_or_raw = 'raw',
        batch_size=5,
        data_augm_kwargs={},
        return_one_hot=False,
        return_01c=False,
        return_list=True,
        use_threads=False)

    train_nsamples = trainiter.nsamples
    train_nbatches = trainiter.nbatches
    valid_nbatches = validiter.nbatches
    valid_nbatches2 = validiter2.nbatches
    print("Train %d" % (train_nsamples))


    # Simulate training
    max_epochs = 2
    start_training = time.time()
    for epoch in range(max_epochs):
        start_epoch = time.time()
        for mb in range(10):
            start_batch = time.time()
            batch = trainiter.next()
            print("Minibatch train {}: {} sec".format(mb, (time.time() -
                                                     start_batch)))
        for mb in range(10):
            start_batch = time.time()
            batch = validiter.next()
            print("Minibatch valid {}: {} sec".format(mb, (time.time() -
                                                     start_batch)))

        for mb in range(10):
            start_batch = time.time()
            batch = validiter2.next()
            print("Minibatch valid {}: {} sec".format(mb, (time.time() -
                                                     start_batch)))



        print("Epoch time: %s" % str(time.time() - start_epoch))
    print("Training time: %s" % str(time.time() - start_training))


def test_parcellation():

    train_iter = ParcellationDataset(
        which_set='train',
        smooth_or_raw = 'both',
        batch_size=5,
        data_augm_kwargs={},
        return_one_hot=False,
        return_01c=False,
        return_list=True,
        use_threads=False)

    valid_iter = ParcellationDataset(
        which_set='valid',
        smooth_or_raw = 'raw',
        batch_size=5,
        data_augm_kwargs={},
        return_one_hot=False,
        return_01c=False,
        return_list=True,
        use_threads=False)

    train_nsamples = train_iter.nsamples
    valid_nsamples = valid_iter.nsamples

    print 'train n samples', train_nsamples
    print 'valid n samples', valid_nsamples



    # Simulate training
    max_epochs = 2
    start_training = time.time()
    for epoch in range(max_epochs):
        start_epoch = time.time()
        for mb in range(2):
            start_batch = time.time()
            batch = train_iter.next()
            print("Minibatch train {}: {} sec".format(mb, (time.time() -
                                                     start_batch)))
        for mb in range(2):
            start_batch = time.time()
            batch = valid_iter.next()
            print("Minibatch valid {}: {} sec".format(mb, (time.time() -
                                                     start_batch)))


        print("Epoch time: %s" % str(time.time() - start_epoch))
    print("Training time: %s" % str(time.time() - start_training))



if __name__ == '__main__':
    #test_6layers()
    test_parcellation()
