import numpy as np
import os

import dataset_loaders
from dataset_loaders.parallel_loader import ThreadedDataset


floatX = 'float32'


class TestDataset(ThreadedDataset):
    name = 'test'
    non_void_nclasses = 10
    _void_labels = []

    # optional arguments
    data_shape = (360, 480, 3)

    _filenames = None

    @property
    def filenames(self):
        return None

    def __init__(self, *args, **kwargs):

        self.path = os.path.join(
            dataset_loaders.__path__[0], 'datasets', 'camvid', 'segnet')
        self.sharedpath = '/data/lisa/exp/visin/_datasets/camvid/segnet'

        # constructing the ThreadedDataset
        # it also creates/copies the dataset in self.path if not already there
        super(TestDataset, self).__init__(*args, **kwargs)

    def get_names(self):
        """Return a dict of names, per prefix/subset."""
        return {'test': ['filename']}

    def load_sequence(self, first_frame):
        """Load a sequence of images/frames

        Auxiliary function that loads a sequence of frames with
        the corresponding ground truth and their filenames.
        Returns a dict with the images in [0, 1], their corresponding
        labels, their subset (i.e. category, clip, prefix) and their
        filenames.
        """
        max_label = self.non_void_nclasses + len(self._void_labels)
        mask = np.array(range(max_label) * 8).reshape((max_label, 8))
        img = np.random.random((max_label, 8, 1))
        print('Mask before remap: \n{}'.format(mask))
        print('Void labels: {}'.format(self._void_labels))
        print('Max: {} \nExpected max after remap: {}'.format(
            mask.max(), self.non_void_nclasses))

        ret = {}
        ret['data'] = np.array([img])
        ret['subset'] = 'default'
        ret['labels'] = np.array([mask])
        ret['filenames'] = ['']
        return ret


class TestDataset2voids(TestDataset):
    non_void_nclasses = 10
    _void_labels = [3, 5]


class TestDataset4voids(TestDataset):
    non_void_nclasses = 9
    _void_labels = [0, 3, 5, 11]


def test_one_hot_mapping(dd, mapping, inv_mapping):
    max_label = dd.non_void_nclasses + len(dd._void_labels)

    # The mask before the remapping
    original_aa = np.array(range(max_label) * 8).reshape((max_label, 8))
    # The mask after the remapping
    aa = dd.next()[1][0]
    print(aa)
    print('Max after remap: {}'.format(aa.max()))

    # i == each element of the original array
    for i in range(dd.non_void_nclasses + len(dd._void_labels)):
        assert all(aa[original_aa == i] == mapping[i]), (
            'Key {} failed. aa content: \n{}'.format(
                i, aa[original_aa == i]))
    for i in range(dd.non_void_nclasses):
        assert all(original_aa[aa == i] == inv_mapping[i]), (
            'Key {} failed. original_aa content: \n{}'.format(
                i, original_aa[aa == i]))
    assert all([el in dd._void_labels
                for el in original_aa[aa == dd.non_void_nclasses]])
    print('Test successful!')
    print('##########################################\n\n')


if __name__ == "__main__":
    # 0 voids
    dd = TestDataset(return_list=True)
    mapping = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9}
    inv_mapping = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9}
    test_one_hot_mapping(dd, mapping, inv_mapping)

    # 2 voids
    dd = TestDataset2voids(return_list=True)
    mapping = {0: 0, 1: 1, 2: 2, 3: 10, 4: 3, 5: 10, 6: 4, 7: 5, 8: 6,
               9: 7, 10: 8, 11: 9}
    inv_mapping = {0: 0, 1: 1, 2: 2, 3: 4, 4: 6, 5: 7, 6: 8, 7: 9, 8: 10,
                   9: 11}
    test_one_hot_mapping(dd, mapping, inv_mapping)

    # 4 voids
    dd = TestDataset4voids(return_list=True)
    mapping = {0: 9, 1: 0, 2: 1, 3: 9, 4: 2, 5: 9, 6: 3, 7: 4, 8: 5,
               9: 6, 10: 7, 11: 9, 12: 8}
    inv_mapping = {0: 1, 1: 2, 2: 4, 3: 6, 4: 7, 5: 8, 6: 9, 7: 10, 8: 12}
    test_one_hot_mapping(dd, mapping, inv_mapping)
