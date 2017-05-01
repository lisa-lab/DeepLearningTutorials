import os
from StringIO import StringIO
import sys
import unittest

import numpy as np

import dataset_loaders
from dataset_loaders.parallel_loader import ThreadedDataset


class TestDataset(ThreadedDataset):
    name = 'Test'
    non_void_nclasses = []
    data_shape = (1, 1, 1, 1)
    _void_labels = []
    path = os.path.join(dataset_loaders.__path__[0], 'datasets', 'camvid',
                        'segnet')
    sharedpath = ''

    def __init__(self, raiseIO=False, *args, **kwargs):
        self.raiseIO = raiseIO
        super(TestDataset, self).__init__(*args, **kwargs)

    def get_names(self):
        return {'default': [1, 2, 3, 4, 5, 6]}

    def load_sequence(self, sequence):
        data = np.array([int(el) for _, el in sequence])
        data = data[:, None, None, None]
        labels = np.array([el for _, el in sequence])
        labels = labels[:, None, None]
        ret = {'data': data,
               'labels': labels,
               'filenames': np.array([el for _, el in sequence])}
        if 6 in data:
            if self.raiseIO:
                # raise IOError
                open('non_existing_file', 'r')
            else:
                raise RuntimeError('Test error')
        return ret


class TestException(unittest.TestCase):
    def testIOError(self, verbose=False):
        for threads in [False, True]:
            dd = TestDataset(use_threads=threads, batch_size=2,
                             shuffle_at_each_epoch=False, raiseIO=True)
            # suppress the text
            out = StringIO()
            sys.stdout = out
            for i in range(12):
                try:
                    aa = dd.next()['data'][:, 0, 0, 0]
                except:
                    self.fail("testIOError failed!")
                if verbose:
                    print("TestIO: Thread {} Minibatch {}: {}".format(
                        threads, i, aa))

    def testRuntimeError(self, verbose=False):
        for threads in [False, True]:
            dd = TestDataset(use_threads=threads, batch_size=2,
                             shuffle_at_each_epoch=False)
            for i in range(12):
                if (i+1) % 3 == 0:
                    with self.assertRaises(RuntimeError):
                        aa = dd.next()
                else:
                    aa = dd.next()['data'][:, 0, 0, 0]
                    if verbose:
                        print("TestRuntime: Thread {} Minibatch {}: {}".format(
                            threads, i, aa))


if __name__ == '__main__':
        unittest.main()
