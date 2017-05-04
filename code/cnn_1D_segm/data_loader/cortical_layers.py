import os
import time

import numpy as np
from PIL import Image
import re
import warnings

from dataset_loaders.parallel_loader import ThreadedDataset
from parallel_loader_1D import ThreadedDataset_1D

floatX = 'float32'

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
        5: (0, 0, 64),         # layer 5
        6: (64, 64, 128),      # layer 6
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
                 smooth_or_raw = 'both',
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

        if smooth_or_raw == 'both':
            self.data_shape = (200,2)
        else :
            self.data_shape = (200,1)

        super(Cortical6LayersDataset, self).__init__(*args, **kwargs)

    def get_names(self):
        """Return a dict of names, per prefix/subset."""

        return {'default': self.filenames}



def test_6layers():
    train_iter = Cortical6LayersDataset(
        which_set='train',
        smooth_or_raw = 'both',
        batch_size=500,
        data_augm_kwargs={},
        return_one_hot=False,
        return_01c=False,
        return_list=True,
        use_threads=False)

    valid_iter = Cortical6LayersDataset(
        which_set='valid',
        smooth_or_raw = 'smooth',
        batch_size=500,
        data_augm_kwargs={},
        return_one_hot=False,
        return_01c=False,
        return_list=True,
        use_threads=False)

    valid_iter2 = Cortical6LayersDataset(
        which_set='valid',
        smooth_or_raw = 'raw',
        batch_size=500,
        data_augm_kwargs={},
        return_one_hot=False,
        return_01c=False,
        return_list=True,
        use_threads=False)



    train_nsamples = train_iter.nsamples
    train_nbatches = train_iter.nbatches
    valid_nbatches = valid_iter.nbatches
    valid_nbatches2 = valid_iter2.nbatches



    # Simulate training
    max_epochs = 1
    print "Simulate training for", str(max_epochs), "epochs"
    start_training = time.time()
    for epoch in range(max_epochs):
        print "Epoch #", str(epoch)

        start_epoch = time.time()

        print "Iterate on the training set", train_nbatches, "minibatches"
        for mb in range(train_nbatches):
            start_batch = time.time()
            batch = train_iter.next()
            if mb%5 ==0:
                print("Minibatch train {}: {} sec".format(mb, (time.time() -
                                                     start_batch)))

        print "Iterate on the validation set", valid_nbatches, "minibatches"
        for mb in range(valid_nbatches):
            start_batch = time.time()
            batch = valid_iter.next()
            if mb%5 ==0:
                print("Minibatch valid {}: {} sec".format(mb, (time.time() -
                                                     start_batch)))

        print "Iterate on the validation set (second time)", valid_nbatches2, "minibatches"
        for mb in range(valid_nbatches2):
            start_batch = time.time()
            batch = valid_iter2.next()
            if mb%5==0:
                print("Minibatch valid {}: {} sec".format(mb, (time.time() -
                                                     start_batch)))

        print("Epoch time: %s" % str(time.time() - start_epoch))
    print("Training time: %s" % str(time.time() - start_training))

if __name__ == '__main__':
    print "Loading the dataset 1 batch at a time"
    test_6layers()
    print "Success!"
