import os
import sys
import time
import warnings

import numpy as np
from PIL import Image

from dataset_loaders.parallel_loader import ThreadedDataset


floatX = 'float32'


class MSCocoDataset(ThreadedDataset):
    '''The Microsoft Common Object in Context (COCO) dataset

    The MS COCO dataset [1]_ consists of more than 300,000 images and over
    2 Million annotated instances.

    This loader only loads the segmentation information, but the other
    annotations could added as well with little effort (PR welcome!).

    The dataset should be downloaded from [1]_ into the `shared_path`
    (that should be specified in the config.ini according to the
    instructions in ../README.md).

    Parameters
    ----------
    which_set: string
        A string in ['train', 'val', 'valid', 'test'], corresponding to the set
        to be returned.
    warn_grayscale: bool
        Some of the images of the dataset are grayscale. If True, the
        loader will raise a warning when a grayscale image is loaded,
        providing the name of the image.

     References
    ----------
    .. [1] http://mscoco.org/
    '''
    name = 'mscoco'
    non_void_nclasses = 80
    _void_labels = [0, 12, 26, 29, 30, 45, 66, 68, 69, 71, 83]

    _filenames = None
    _image_path = None
    _coco = None

    @property
    def prefix_list(self):
        return self.coco.getCatIds()

    @property
    def filenames(self):
        if self._filenames is None:
            self._filenames = self.coco.getImgIds()
        return self._filenames

    @property
    def image_path(self):
        if self._image_path is None:
            if self.which_set == 'train':
                self._image_path = os.path.join(self.path, 'images',
                                                'train2014')
            elif self.which_set == 'val':
                self._image_path = os.path.join(self.path, 'images',
                                                'val2014')
            elif self.which_set == 'test':
                self._image_path = os.path.join(self.path, 'images',
                                                'test2015')
        return self._image_path

    @property
    def coco(self):
        if self._coco is None:
            from pycocotools.coco import COCO
            if self.which_set == 'train':
                self._coco = COCO(os.path.join(self.path, 'annotations',
                                               'instances_train2014.json'))
            elif self.which_set == 'val':
                self._coco = COCO(os.path.join(self.path, 'annotations',
                                               'instances_val2014.json'))
            elif self.which_set == 'test':
                self._coco = COCO(os.path.join(self.path, 'annotations',
                                               'image_info_test2015.json'))
        return self._coco

    def __init__(self,
                 which_set='train',
                 warn_grayscale=False,
                 *args,
                 **kwargs):
        sys.path.append(os.path.join(os.path.dirname(
            os.path.abspath(__file__)), 'coco', 'PythonAPI'))
        self.which_set = 'val' if which_set == 'valid' else which_set
        self.warn_grayscale = warn_grayscale

        # constructing the ThreadedDataset
        # it also creates/copies the dataset in self.path if not already there
        super(MSCocoDataset, self).__init__(*args, **kwargs)

        self.set_has_GT = self.which_set != 'test'

        if self.seq_length != 1 or self.seq_per_subset != 0:
            raise NotImplementedError('Images in COCO are not sequential. '
                                      'It does not make sense to request a '
                                      'sequence. seq_length {} '
                                      'seq_per_subset {}'.format(
                                          self.seq_length,
                                          self.seq_per_subset))

    def get_names(self):
        """Return a dict of names, per prefix/subset."""
        per_subset_names = {}
        coco = self.coco
        # Populate self.prefix_list
        prefix_list = self.prefix_list

        # cycle through the different categories
        for prefix in prefix_list:
            per_subset_names[prefix] = coco.loadImgs(
                coco.getImgIds(catIds=prefix))
        return per_subset_names

    def load_sequence(self, sequence):
        """Load a sequence of images/frames

        Auxiliary function that loads a sequence of frames with
        the corresponding ground truth and their filenames.
        Returns a dict with the images in [0, 1], their corresponding
        labels, their subset (i.e. category, clip, prefix) and their
        filenames.
        """
        from pycocotools import mask as cocomask
        from matplotlib.path import Path
        X = []
        Y = []
        F = []

        for prefix, img in sequence:
            if not os.path.exists('%s/%s' % (self.image_path,
                                             img['file_name'])):
                raise RuntimeError('Image %s is missing' % img['file_name'])

            im = Image.open('%s/%s' % (self.image_path,
                                       img['file_name'])).copy()
            if im.mode == 'L':
                if self.warn_grayscale:
                    warnings.warn('image %s is grayscale..' % img['file_name'],
                                  RuntimeWarning)
                im = im.convert('RGB')

            # load the annotations and build the mask
            anns = self.coco.loadAnns(self.coco.getAnnIds(
                    imgIds=img['id'], catIds=prefix, iscrowd=None))

            mask = np.zeros(im.size).transpose(1, 0)
            for ann in anns:
                catId = ann['category_id']
                if type(ann['segmentation']) == list:
                    # polygon
                    for seg in ann['segmentation']:
                        # xy vertex of the polygon
                        poly = np.array(seg).reshape((len(seg)/2, 2))
                        closed_path = Path(poly)
                        nx, ny = img['width'], img['height']
                        x, y = np.meshgrid(np.arange(nx), np.arange(ny))
                        x, y = x.flatten(), y.flatten()
                        points = np.vstack((x, y)).T
                        grid = closed_path.contains_points(points)
                        if np.count_nonzero(grid) == 0:
                            warnings.warn(
                                'One of the annotations that compose the mask '
                                'of %s was empty' % img['file_name'],
                                RuntimeWarning)
                        grid = grid.reshape((ny, nx))
                        mask[grid] = catId
                else:
                    # mask
                    if type(ann['segmentation']['counts']) == list:
                        rle = cocomask.frPyObjects(
                            [ann['segmentation']],
                            img['height'], img['width'])
                    else:
                        rle = [ann['segmentation']]
                    grid = cocomask.decode(rle)[:, :, 0]
                    grid = grid.astype('bool')
                    mask[grid] = catId

            mask = np.array(mask.astype('int32'))
            im = np.array(im).astype(floatX) / 255.
            X.append(im)
            Y.append(mask)
            F.append(img['file_name'])

        ret = {}
        ret['data'] = np.array(X)
        ret['labels'] = np.array(Y)
        ret['subset'] = prefix
        ret['filenames'] = np.array(F)
        return ret


def test():
    trainiter = MSCocoDataset(
        which_set='train',
        batch_size=5,
        seq_per_subset=0,
        seq_length=0,
        data_augm_kwargs={
            'crop_size': (72, 59)},
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

    print('Total number of batches: %d' % nbatches)
    for epoch in range(max_epochs):
        for mb in range(nbatches):
            train_group = trainiter.next()

            # train_group checks
            assert train_group[0].ndim == 4
            assert train_group[0].shape[0] <= train_batch_size
            assert train_group[0].shape[1:] == (72, 59, 3)
            assert train_group[0].min() >= 0
            assert train_group[0].max() <= 1
            assert train_group[1].ndim == 4
            assert train_group[1].shape[0] <= train_batch_size
            assert train_group[1].shape[1:] == (72, 59, nclasses)
            # time.sleep approximates running some model
            time.sleep(0.1)
            stop = time.time()
            part = stop - start - 0.1
            start = stop
            tot += part
            print("Minibatch %s time: %s (%s)" % (str(mb), part, tot))


def run_tests():
    test()


if __name__ == '__main__':
    run_tests()
