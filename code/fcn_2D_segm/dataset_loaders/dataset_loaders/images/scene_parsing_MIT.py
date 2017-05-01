import os
import time

import numpy as np
from PIL import Image

from dataset_loaders.parallel_loader import ThreadedDataset
floatX = 'float32'


# Load file with the description of the dataset classes
def load_class_names(file_name):
    # Read the csv data
    from numpy import genfromtxt
    csv_data = genfromtxt(file_name, delimiter='\t', dtype=None, skip_header=1)
    # print str(csv_data)

    # Create the mask labels dictionary
    mask_labels = {}
    for line in csv_data:
        mask_labels[int(line[0])] = line[4]
    # print(str(mask_labels))

    # TODO: Other data can be taken from here like class frequency

    return mask_labels


class SceneParsingMITDataset(ThreadedDataset):
    '''The MIT Scene Parsing benchmark (SceneParse150)

    SceneParse150 [1]_ provides a standard training and evaluation platform for
    scene parsing algorithms. The data for this benchmark comes from ADE20K
    Dataset, which contains more than 20K scene-centric images exhaustively
    annotated with objects and object parts. The dataset contains 20K annotated
    images for training and 2K annotated images for validation. The test set is
    a separate batch of images, for which segmentation labels are not provided.
    The ground truth labels associate each pixel with one of the 150 semantic
    classes, plus one void class.

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
    .. [1] http://sceneparsing.csail.mit.edu/
    '''
    name = 'scene_parsing_MIT'
    non_void_nclasses = 150
    _void_labels = [0]

    GTclasses = range(non_void_nclasses) + _void_labels

    _filenames = None

    @property
    def filenames(self):
        import glob

        if self._filenames is None:
            # Load filenames
            filenames = []

            # Get file names from images folder
            file_pattern = os.path.join(self.image_path, "*.jpg")
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

    def __init__(self, which_set="train", *args, **kwargs):

        # Put which_set in canonical form:training, validation, testing
        if which_set in ("train", "training"):
            self.which_set = "training"
        elif which_set in ("val", "valid", "validation"):
            self.which_set = "validation"
        elif which_set in ("test", "testing"):
            self.which_set = "testing"
            self.set_has_GT = False
        else:
            raise ValueError("Unknown set requested: %s" % which_set)

        # Define the txt, images and mask paths
        self.txt_path = os.path.join(self.path, "objectInfo150.txt")
        self.image_path = os.path.join(self.path, "images", self.which_set)
        self.mask_path = os.path.join(self.path, "annotations", self.which_set)

        # Load info from the classes
        # _mask_labels = load_class_names(self.txt_path)

        super(SceneParsingMITDataset, self).__init__(*args, **kwargs)

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
        image_batch = []
        mask_batch = []
        filename_batch = []

        for prefix, img_name in sequence:

            # Load image
            img = io.imread(os.path.join(self.image_path, img_name + ".jpg"))
            img = img.astype(floatX) / 255.

            # Load mask
            if self.set_has_GT:
                mask = np.array(Image.open(
                    os.path.join(self.mask_path, img_name + ".png")))
                mask = mask.astype('int32')
            else:
                mask = []

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
    trainiter = SceneParsingMITDataset(
        which_set='train',
        batch_size=100,
        seq_per_subset=0,
        seq_length=0,
        data_augm_kwargs={
            'crop_size': (224, 224)},
        return_one_hot=True,
        return_01c=True,
        return_list=True,
        use_threads=False)

    validiter = SceneParsingMITDataset(
        which_set='valid',
        batch_size=5,
        seq_per_subset=0,
        seq_length=0,
        data_augm_kwargs={
            'crop_size': (224, 224)},
        return_one_hot=True,
        return_01c=True,
        return_list=True,
        use_threads=False)

    testiter = SceneParsingMITDataset(
        which_set='test',
        batch_size=5,
        seq_per_subset=0,
        seq_length=0,
        data_augm_kwargs={
            'crop_size': (224, 224)},
        return_one_hot=True,
        return_01c=True,
        return_list=True,
        use_threads=False)

    # Get number of classes
    nclasses = trainiter.nclasses
    print ("N classes: " + str(nclasses))

    # Training info
    train_nsamples = trainiter.nsamples
    train_batch_size = trainiter.batch_size
    train_nbatches = trainiter.nbatches
    print("Train n_images: {}, batch_size{}, n_batches{}".format(
        train_nsamples, train_batch_size, train_nbatches))

    # Validation info
    valid_nsamples = validiter.nsamples
    valid_batch_size = validiter.batch_size
    valid_nbatches = validiter.nbatches
    print("Train n_images: {}, batch_size: {}, n_batches: {}".format(
        valid_nsamples, valid_batch_size, valid_nbatches))

    # Testing info
    test_nsamples = testiter.nsamples
    test_batch_size = testiter.batch_size
    test_nbatches = testiter.nbatches
    print("Test n_images: {}, batch_size: {}, n_batches: {}".format(
        test_nsamples, test_batch_size, test_nbatches))

    max_epochs = 2

    for epoch in range(max_epochs):
        epoch_start = time.time()
        for mb in range(train_nbatches):
            mb_start = time.time()
            trainiter.next()
            print("Minibatch {}: {:.3f} sec".format(mb, time.time() -
                                                    mb_start))
        print("End epoch {}: {:.3f} sec".format(epoch, time.time() -
                                                epoch_start))


def run_tests():
    test()


if __name__ == '__main__':
    run_tests()
