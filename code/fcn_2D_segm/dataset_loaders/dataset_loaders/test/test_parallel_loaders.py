import time
import unittest

from dataset_loaders import CamvidDataset


class TestParallelLoaders(unittest.TestCase):

    def testShapes5D(self, verbose=False):
        trainiter = CamvidDataset(
            which_set='train',
            batch_size=5,
            seq_per_subset=0,
            seq_length=10,
            data_augm_kwargs={
                'crop_size': (224, 224)},
            return_one_hot=True,
            return_01c=True,
            return_list=True,
            use_threads=True,
            nthreads=5)

        train_nsamples = trainiter.nsamples
        nclasses = trainiter.nclasses
        nbatches = trainiter.nbatches
        train_batch_size = trainiter.batch_size
        print("Train %d" % (train_nsamples))

        start = time.time()
        tot = 0
        max_epochs = 5

        for epoch in range(max_epochs):
            for mb in range(nbatches):
                train_group = trainiter.next()
                if train_group is None:
                    raise RuntimeError('One batch was missing')

                # train_group checks
                self.assertEqual(train_group[0].ndim, 5)
                self.assertLessEqual(train_group[0].shape[0], train_batch_size)
                self.assertEqual(train_group[0].shape[1:], (10, 224, 224, 3))
                self.assertGreaterEqual(train_group[0].min(), 0)
                self.assertLessEqual(train_group[0].max(), 1)
                self.assertEqual(train_group[1].ndim, 5)
                self.assertLessEqual(train_group[1].shape[0],
                                     train_batch_size)
                self.assertEqual(train_group[1].shape[1:],
                                 (10, 224, 224, nclasses))

                if verbose:
                    # time.sleep approximates running some model
                    time.sleep(0.5)
                    stop = time.time()
                    part = stop - start - 1
                    start = stop
                    tot += part
                    print("Minibatch %s time: %s (%s)" % (str(mb), part, tot))

    def testShapes4D(self, verbose=False):
        trainiter = CamvidDataset(
            which_set='train',
            batch_size=5,
            seq_per_subset=0,
            seq_length=0,
            data_augm_kwargs={
                'crop_size': (224, 224)},
            return_one_hot=True,
            return_01c=True,
            return_list=True,
            use_threads=True,
            nthreads=5)

        train_nsamples = trainiter.nsamples
        nclasses = trainiter.nclasses
        nbatches = trainiter.nbatches
        train_batch_size = trainiter.batch_size
        print("Train %d" % (train_nsamples))

        start = time.time()
        tot = 0
        max_epochs = 5

        for epoch in range(max_epochs):
            for mb in range(nbatches):
                train_group = trainiter.next()

                # train_group checks
                self.assertEqual(train_group[0].ndim, 4)
                self.assertLessEqual(train_group[0].shape[0], train_batch_size)
                self.assertEqual(train_group[0].shape[1], 224)
                self.assertEqual(train_group[0].shape[2], 224)
                self.assertEqual(train_group[0].shape[3], 3)
                self.assertGreaterEqual(train_group[0].min(), 0)
                self.assertLessEqual(train_group[0].max(), 1)
                self.assertEqual(train_group[1].ndim, 4)
                self.assertLessEqual(train_group[1].shape[0], train_batch_size)
                self.assertEqual(train_group[1].shape[1], 224)
                self.assertEqual(train_group[1].shape[2], 224)
                self.assertEqual(train_group[1].shape[3], nclasses)

                # time.sleep approximates running some model
                time.sleep(0.5)
                if verbose:
                    stop = time.time()
                    part = stop - start - 1
                    start = stop
                    tot += part
                    print("Minibatch %s time: %s (%s)" % (str(mb), part, tot))


if __name__ == '__main__':
        unittest.main()
