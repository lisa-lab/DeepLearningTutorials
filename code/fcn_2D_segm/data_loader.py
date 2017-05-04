import os
import time

from dataset_loaders.images.polyps912 import Polyps912Dataset
from dataset_loaders.images.camvid import CamvidDataset
from dataset_loaders.images.polyps912 import Polyps912Dataset
from dataset_loaders.images.isbi_em_stacks import IsbiEmStacksDataset


def load_data(dataset, train_data_augm_kwargs={}, one_hot=False,
              batch_size=[10, 10, 10], shuffle_train=True, return_0_255=False,
              which_set='all'):

    assert which_set in ['all', 'train', 'val', 'test']

    # Build dataset iterator
    if dataset == 'polyps':
        train_iter = Polyps912Dataset(which_set='train',
                                      batch_size=batch_size[0],
                                      seq_per_subset=0,
                                      seq_length=0,
                                      data_augm_kwargs=train_data_augm_kwargs,
                                      return_one_hot=one_hot,
                                      return_01c=False,
                                      overlap=0,
                                      use_threads=False,
                                      shuffle_at_each_epoch=shuffle_train,
                                      return_list=True,
                                      return_0_255=return_0_255)
        val_iter = Polyps912Dataset(which_set='val',
                                    batch_size=batch_size[1],
                                    seq_per_subset=0,
                                    seq_length=0,
                                    return_one_hot=one_hot,
                                    return_01c=False,
                                    overlap=0,
                                    use_threads=False,
                                    shuffle_at_each_epoch=False,
                                    return_list=True,
                                    return_0_255=return_0_255)
        test_iter = Polyps912Dataset(which_set='test',
                                     batch_size=batch_size[2],
                                     seq_per_subset=0,
                                     seq_length=0,
                                     return_one_hot=one_hot,
                                     return_01c=False,
                                     overlap=0,
                                     use_threads=False,
                                     shuffle_at_each_epoch=False,
                                     return_list=True,
                                     return_0_255=return_0_255)

    elif dataset == 'em':
        train_data_augm_kwargs = {'rotation_range':25,
                             'shear_range':0.41,
                             'horizontal_flip':True,
                             'vertical_flip':True,
                             'fill_mode':'reflect',
                             'spline_warp':True,
                             'warp_sigma':10,
                             'warp_grid_size':3}

        train_iter = IsbiEmStacksDataset(which_set='train',
                                         batch_size=batch_size[0],
                                         seq_per_subset=0,
                                         seq_length=0,
                                         data_augm_kwargs=train_data_augm_kwargs,
                                         return_one_hot=one_hot,
                                         return_01c=False,
                                         overlap=0,
                                         use_threads=True,
                                         shuffle_at_each_epoch=shuffle_train,
                                         return_list=True,
                                         return_0_255=return_0_255)

        val_iter = IsbiEmStacksDataset(which_set='val',
                                       batch_size=batch_size[1],
                                       seq_per_subset=0,
                                       seq_length=0,
                                       return_one_hot=one_hot,
                                       return_01c=False,
                                       use_threads=True,
                                       shuffle_at_each_epoch=False,
                                       return_list=True,
                                       return_0_255=return_0_255)
        test_iter = None
    else:
        print 'Dataset must be either "em" or "polyps" '
        raise NotImplementedError

    if which_set == 'train':
        ret = train_iter
    elif which_set == 'val':
        ret = val_iter
    elif which_set == 'test':
        ret = test_iter
    else:
        ret = [train_iter, val_iter, test_iter]

    return ret




def test_load_em():

    train_iter = IsbiEmStacksDataset(
        which_set='train',
        batch_size=1,
        data_augm_kwargs={},
        return_one_hot=False,
        return_01c=False,
        return_list=True,
        use_threads=False)

    valid_iter = IsbiEmStacksDataset(
        which_set='valid',
        batch_size=1,
        data_augm_kwargs={},
        return_one_hot=False,
        return_01c=False,
        return_list=True,
        use_threads=False)

    test_iter = None

    train_nbatches = train_iter.nbatches
    valid_nbatches = valid_iter.nbatches


    # Simulate training
    max_epochs = 2
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

        print("Epoch time: %s" % str(time.time() - start_epoch))
    print("Training simulation time: %s" % str(time.time() - start_training))


def test_load_polyps():
        train_iter = Polyps912Dataset(
            which_set='train',
            batch_size=1,
            data_augm_kwargs={},
            return_one_hot=False,
            return_01c=False,
            return_list=True,
            use_threads=False)

        valid_iter = Polyps912Dataset(
            which_set='valid',
            batch_size=1,
            data_augm_kwargs={},
            return_one_hot=False,
            return_01c=False,
            return_list=True,
            use_threads=False)

        test_iter = Polyps912Dataset(
            which_set='test',
            batch_size=1,
            data_augm_kwargs={},
            return_one_hot=False,
            return_01c=False,
            return_list=True,
            use_threads=False)

        train_nbatches = train_iter.nbatches
        valid_nbatches = valid_iter.nbatches
        test_nbatches = test_iter.nbatches


        # Simulate training
        max_epochs = 2
        print "Simulate training for", str(max_epochs), "epochs"
        start_training = time.time()
        for epoch in range(max_epochs):
            print "Epoch #", str(epoch)

            start_epoch = time.time()

            print "Iterate on the training set", train_nbatches, "minibatches"
            for mb in range(train_nbatches):
                start_batch = time.time()
                batch = train_iter.next()
                if mb%50 ==0:
                    print("Minibatch train {}: {} sec".format(mb, (time.time() - start_batch)))

            print "Iterate on the validation set", valid_nbatches, "minibatches"
            for mb in range(valid_nbatches):
                start_batch = time.time()
                batch = valid_iter.next()
                if mb%50 ==0:
                    print("Minibatch valid {}: {} sec".format(mb, (time.time() -start_batch)))

            print("Epoch time: %s" % str(time.time() - start_epoch))
        print "Iterate on the test set", test_nbatches, "minibatches"
        for mb in range(test_nbatches):
            start_batch = time.time()
            batch = test_iter.next()
            if mb%50 ==0:
                print("Minibatch test {}: {} sec".format(mb, (time.time() - start_batch)))
        print("Training simulation time: %s" % str(time.time() - start_training))

if __name__=='__main__':
    print "Iterating through  polyps dataset"
    test_load_polyps()
    print "Success!"

    print "Iterating through IsbiEmStacks dataset"
    test_load_em()
    print "Success!"
