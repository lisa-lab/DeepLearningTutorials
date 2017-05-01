import ConfigParser
import os
from os.path import realpath
try:
    import Queue
except ImportError:
    import queue as Queue
import shutil
import sys
from threading import Thread
from time import sleep
import weakref

import numpy as np
from numpy.random import RandomState
from dataset_loaders.data_augmentation import random_transform

import dataset_loaders
from utils_parallel_loader import classproperty, grouper, overlap_grouper


class ThreadedDataset(object):
    _wait_time = 0.05
    __version__ = '1'
    """
    Threaded dataset.

    This is an abstract class and should not be used as is. Each
    specific dataset class should implement its `get_names` and
    `load_sequence` functions to load the list of filenames to be
    loaded and define how to load the data from the dataset,
    respectively.

    See `example_dataset.py` for an example on how to implement a
    specific instance of a dataset.

    Parameters
    ----------
    seq_per_subset: int
        The *maximum* number of sequences per each subset (a.k.a. prefix
        or video). If 0, all sequences will be used. If greater than 0
        and `shuffle_at_each_epoch` is True, at each epoch a new
        selection of sequences per subset will be randomly picked. Default: 0.
    seq_length: int
        The number of frames per sequence. If 0, 4D arrays will be
        returned (not a sequence), else 5D arrays will be returned.
        Default: 0.
    overlap: int
        The number of frames of overlap between the first frame of one
        sample and the first frame of the next. Note that a negative
        overlap will instead specify the number of frames that are
        *skipped* between the last frame of one sample and the first
        frame of the next. None is equivalent to seq_length - 1.
        Default: None.
    batch_size: int
        The size of the batch.
    queues_size: int
        The size of the buffers used in the threaded case. Default: 50.
    return_one_hot: bool
        If True the labels will be returned in one-hot format, i.e. as
        an array of `nclasses` elements all set to 0 except from the id
        of the correct class which is set to 1. Default: False.
    return_01c: bool
        If True the last axis will be the channel axis (01c format),
        else the channel axis will be the third to last (c01 format).
        Default: False.
    return_extended_sequences:bool
        If True the first and last sequence of a batch will be extended so that
        the first frame is repeated `seq_length/2` times. This is useful
        to perform middle frame prediction, i.e., where the current
        frame has to be the middle one and the previous and next ones
        are used as context. Default:False.
    return_middle_frame_only:bool
        If True only the middle frame of the ground truth will be returned.
        Default:False.
    return_0_255: bool
        If True the images will be returned in the range [0, 255] with
        dtype `uint8`. Otherwise the images will be returned in the
        range [0, 1] as dtype `float32`. Default: False.
    use_threads: bool
        If True threads will be used to fetch the data from the dataset.
        Default: False.
    nthreads: int
        The number of threads to use when `use_threads` is True. Default: 1.
    shuffle_at_each_epoch: bool
        If True, at the end of each epoch a new set of batches will be
        prepared and shuffled. Default: True.
    infinite_iterator: bool
        If False a `StopIteration` exception will be raised at the end of an
        epoch. If True no exception will be raised and the dataset will
        behave as an infinite iterator. Default: True.
    return_list: bool
        If True, each call to `next()` will return a list of two numpy arrays
        containing the data and the labels respectively. If False, the
        dataset will instead return a dictionary with the following
        keys:
            * `data`: the augmented/cropped sequence/image
            * `labels`: the corresponding potentially cropped labels
            * `filenames`: the filenames of the frames/images
            * `subset`: the name of the subset the sequence/image belongs to
            * `raw_data`: the original unprocessed sequence/image
        Depending on the dataset, additional keys might be available.
        Default: False.
    data_augm_kwargs: dict
        A dictionary of arguments to be passed to the data augmentation
        function. Default: no data augmentation.
    remove_mean: bool
        If True, the statistics computed dataset-wise will be used to
        remove the dataset mean from the data. Default: False.
    divide_by_std: bool
        If True, the statistics computed dataset-wise will be used to
        divide the data by the dataset standard deviation. Default: False.
    remove_per_img_mean: bool
        If True, each image will be processed to have zero-mean.
        Default: False.
    divide_by_per_img_std=False
        If True, each image will be processed to have unit variance.
        Default: False.
    raise_IOErrors: bool
        If False in case of an IOError a message will be printed on
        screen but no Exception will be raised. Default: False.
    rng: :class:`numpy.random.RandomState` instance
        The random number generator to use. If None, one will be created.
        Default: None.

    Notes
    -----
    The parallel loader will automatically map all non-void classes to be
    sequential starting from 0 and then map all void classes to the
    next class. E.g., suppose non_void_nclasses = 4 and _void_classes = [3, 5]
    the non-void classes will be mapped to 0, 1, 2, 3 and the void
    classes will be mapped to 4, as follows:
        0 --> 0
        1 --> 1
        2 --> 2
        3 --> 4
        4 --> 3
        5 --> 4

    Note also that in case the original labels are not sequential, it
    suffices to list all the original labels as a list in GTclasses for
    parallel_loader to map the non-void classes sequentially starting
    from 0 and all the void classes to the next class. E.g. suppose
    non_void_nclasses = 5, GTclasses = [0, 2, 5, 9, 11, 12, 99] and
    _void_labels = [2, 99], then this will be the mapping:
         0 --> 0
         2 --> 5
         5 --> 1
         9 --> 2
        11 --> 3
        12 --> 4
        99 --> 5
    """
    def __init__(self,
                 seq_per_subset=0,   # if 0 all sequences (or frames, if 4D)
                 seq_length=0,      # if 0, return 4D
                 overlap=None,
                 batch_size=1,
                 queues_size=20,
                 return_one_hot=False,
                 return_01c=False,
                 return_extended_sequences=False,
                 return_middle_frame_only=False,
                 return_0_255=False,
                 use_threads=False,
                 nthreads=1,
                 shuffle_at_each_epoch=True,
                 infinite_iterator=True,
                 return_list=False,  # for keras, return X,Y only
                 data_augm_kwargs={},
                 remove_mean=False,  # dataset stats
                 divide_by_std=False,  # dataset stats
                 remove_per_img_mean=False,  # img stats
                 divide_by_per_img_std=False,  # img stats
                 raise_IOErrors=False,
                 rng=None,
                 **kwargs):

        if len(kwargs):
            print('Unknown arguments: {}'.format(kwargs.keys()))

        # Set default values for the data augmentation params if not specified
        default_data_augm_kwargs = {
            'crop_size': None,
            'rotation_range': 0,
            'width_shift_range': 0,
            'height_shift_range': 0,
            'shear_range': 0,
            'zoom_range': 0,
            'channel_shift_range': 0,
            'fill_mode': 'nearest',
            'cval': 0,
            'cvalMask': 0,
            'horizontal_flip': False,
            'vertical_flip': False,
            'rescale': None,
            'spline_warp': False,
            'warp_sigma': 0.1,
            'warp_grid_size': 3,
            'gamma': 0,
            'gain': 1}

        default_data_augm_kwargs.update(data_augm_kwargs)
        self.data_augm_kwargs = default_data_augm_kwargs
        del(default_data_augm_kwargs, data_augm_kwargs)

        # Put crop_size into canonical form [c1, 2]
        cs = self.data_augm_kwargs['crop_size']
        if cs is not None:
            # Convert to list
            if isinstance(cs, int):
                cs = [cs, cs]
            elif isinstance(cs, tuple):
                cs = list(cs)
            # set 0, 0 to None
            if cs == [0, 0]:
                cs = None
            self.data_augm_kwargs['crop_size'] = cs

        # Do not support multithread without shuffling
        if use_threads and nthreads > 1 and not shuffle_at_each_epoch:
            raise NotImplementedError('Multiple threads are not order '
                                      'preserving')

        # Check that the implementing class has all the mandatory attributes
        mandatory_attrs = ['name', 'non_void_nclasses', '_void_labels']
        missing_attrs = [attr for attr in mandatory_attrs if not
                         hasattr(self, attr)]
        if missing_attrs != []:
            raise NameError('Mandatory argument(s) missing: {}'.format(
                missing_attrs))
        if hasattr(self, 'GT_classes'):
            raise NameError('GTclasses mispelled as GT_classes')

        # If variable sized dataset --> either batch_size 1 or crop
        if (not hasattr(self, 'data_shape') and batch_size > 1 and
                not self.data_augm_kwargs['crop_size']):
            raise ValueError(
                '{} has no `data_shape` attribute, this means that the '
                'shape of the samples varies across the dataset. You '
                'must either set `batch_size = 1` or specify a '
                '`crop_size`'.format(self.name))

        if seq_length and overlap and overlap >= seq_length:
            raise ValueError('`overlap` should be smaller than `seq_length`')

        # Copy the data to the local path if not existing
        if not os.path.exists(self.path):
            print('The local path {} does not exist. Copying '
                  'the dataset...'.format(self.path))
            shutil.copytree(self.shared_path, self.path)
            for r,d,f in os.walk(self.path):
                os.chmod(r,0775)
            print('Done.')
        else:
            try:
                with open(os.path.join(self.path, '__version__')) as f:
                    if f.read() != self.__version__:
                        raise IOError
            except IOError:
                print('The local path {} exist, but is outdated. I will '
                      'replace the old files with the new ones...'.format(
                          self.path))
                if not os.path.exists(self.shared_path):
                    print('The shared_path {} for {} does not exist. Please '
                          'edit the config.ini file with a valid path, as '
                          'specified in the README.'.format(self.shared_path,
                                                            self.name))
                if realpath(self.path) != realpath(self.shared_path):
                    shutil.rmtree(self.path)
                    shutil.copytree(self.shared_path, self.path)
                    for r,d,f in os.walk(self.path):
                        os.chmod(r,0775)
                with open(os.path.join(self.path, '__version__'), 'w') as f:
                    f.write(self.__version__)
                print('Done.')

        # Save parameters in object
        self.seq_per_subset = seq_per_subset
        self.return_sequence = seq_length != 0
        self.seq_length = seq_length if seq_length else 1
        self.overlap = overlap if overlap is not None else self.seq_length - 1
        self.batch_size = batch_size
        self.queues_size = queues_size
        self.return_one_hot = return_one_hot
        self.return_01c = return_01c
        self.return_extended_sequences = return_extended_sequences
        self.return_middle_frame_only = return_middle_frame_only
        self.return_0_255 = return_0_255
        self.use_threads = use_threads
        self.nthreads = nthreads
        self.shuffle_at_each_epoch = shuffle_at_each_epoch
        self.infinite_iterator = infinite_iterator
        self.return_list = return_list
        self.remove_mean = remove_mean
        self.divide_by_std = divide_by_std
        self.remove_per_img_mean = remove_per_img_mean
        self.divide_by_per_img_std = divide_by_per_img_std
        self.raise_IOErrors = raise_IOErrors
        self.rng = rng if rng is not None else RandomState(0xbeef)

        self.set_has_GT = getattr(self, 'set_has_GT', True)
        self.mean = getattr(self, 'mean', [])
        self.std = getattr(self, 'std', [])

        # ...01c
        data_shape = list(getattr(self.__class__, 'data_shape',
                                  (None, None, 3)))
        if self.data_augm_kwargs['crop_size']:
            data_shape[-3:-1] = self.data_augm_kwargs['crop_size']  # change 01
        if self.return_01c:
            self.data_shape = data_shape
        else:
            self.data_shape = [data_shape[i] for i in
                               [2] + range(2) + range(3, len(data_shape))]

        # Load a dict of names, per video/subset/prefix/...
        self.names_per_subset = self.get_names()

        # Fill the sequences/batches lists and initialize everything
        self._fill_names_sequences()
        if len(self.names_sequences) == 0:
            raise RuntimeError('The name list cannot be empty')
        self._fill_names_batches(shuffle_at_each_epoch)

        if self.use_threads:
            # Initialize the queues
            self.names_queue = Queue.Queue(maxsize=self.queues_size)
            self.data_queue = Queue.Queue(maxsize=self.queues_size)
            self._init_names_queue()  # Fill the names queue

            # Start the data fetcher threads
            self.sentinel = object()  # guaranteed unique reference
            self.data_fetchers = []
            for _ in range(self.nthreads):
                data_fetcher = Thread(
                    target=threaded_fetch,
                    args=(weakref.ref(self),))
                data_fetcher.setDaemon(True)  # Die when main dies
                data_fetcher.start()
                data_fetcher = weakref.ref(data_fetcher)
                self.data_fetchers.append(data_fetcher)
            # Give time to the data fetcher to die, in case of errors
            # sleep(1)

    def get_names(self):
        """ Loads ALL the names, per video.

        Should return a *dictionary*, where each element of the
        dictionary is a list of filenames. The keys of the dictionary
        should be the prefixes, i.e., names of the subsets of the
        dataset. If the dataset has no subset, 'default' can be used as
        a key.
        """
        raise NotImplementedError

    def load_sequence(self, sequence):
        """ Loads a 4D sequence from the dataset.

        Should return a *dict* with at least these keys:
            * 'data': the images or frames of the sequence
            * 'labels': the labels of the sequence
            * 'subset': the subset/clip/category/.. the sequence belongs to
            * 'filenames': the filenames of each image/frame of the sequence
        """
        raise NotImplementedError

    def _fill_names_sequences(self):
        names_sequences = {}

        # Cycle over prefix/subset/video/category/...
        for prefix, names in self.names_per_subset.items():
            seq_length = self.seq_length

            # Repeat the first and last elements so that the first and last
            # sequences are filled with repeated elements up/from the
            # middle element.
            if self.return_extended_sequences:
                names = ([names[0]] * (seq_length // 2) + names +
                         [names[-1]] * (seq_length // 2))
            # Fill sequences with multiple el with form
            # [(prefix, name1), (prefix, name2), ...]. The names here
            # have overlap = (seq_length - 1), i.e., "stride" = 1
            sequences = [el for el in overlap_grouper(
                names, seq_length, prefix=prefix)]
            # Sequences of frames with the requested overlap
            sequences = sequences[::self.seq_length - self.overlap]

            names_sequences[prefix] = sequences
        self.names_sequences = names_sequences

    def _fill_names_batches(self, shuffle):
        '''Create the desired batches of sequences

        * Select the desired sequences according to the parameters
        * Set self.nsamples, self.nbatches and self.names_batches.
        * Set self.names_batches, an iterator over the batches of names.
        '''
        names_sequences = []
        for prefix, sequences in self.names_sequences.items():

            # Pick only a subset of sequences per each video
            if self.seq_per_subset:
                # Pick `seq_per_subset` random indices
                idx = np.random.permutation(range(len(sequences)))[
                    :self.seq_per_subset]
                # Select only those sequences
                sequences = np.array(sequences)[idx]

            names_sequences.extend(sequences)

        # Shuffle the sequences
        if shuffle:
            self.rng.shuffle(names_sequences)

        # Group the sequences into minibatches
        names_batches = [el for el in grouper(names_sequences,
                                              self.batch_size)]
        self.nsamples = len(names_sequences)
        self.nbatches = len(names_batches)
        self.names_batches = iter(names_batches)

    def _init_names_queue(self):
        for _ in range(self.queues_size):
            try:
                name_batch = self.names_batches.next()
                self.names_queue.put(name_batch)
            except StopIteration:
                # Queue is bigger than the tot number of batches
                break

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        return self._step()

    def _step(self):
        '''Return one batch

        In case of threading, get one batch from the `data_queue`.
        The infinite loop allows to wait for the fetchers if data is
        consumed too fast.
        '''
        done = False
        while not done:
            if self.use_threads:
                # THREADS
                # Kill main process if fetcher died
                if all([df() is None or not df().isAlive()
                        for df in self.data_fetchers]):
                    import sys
                    print('All fetchers threads died. I will suicide!')
                    sys.exit(0)
                try:
                    # Get one minibatch from the out queue
                    data_batch = self.data_queue.get(False)
                    self.data_queue.task_done()
                    # Exception handling
                    if len(data_batch) == 3:
                        if (isinstance(data_batch[1], IOError) and not
                                self.raise_IOErrors):
                            print('WARNING: Image corrupted or missing!')
                            print(data_batch[1])
                            continue  # fetch the next element
                        if (isinstance(data_batch[1], type(BaseException)) or
                                isinstance(data_batch[1], BaseException)):
                            raise data_batch[0], data_batch[1], data_batch[2]
                    done = True
                    # Refill the names queue, if we still have batches
                    try:
                        name_batch = self.names_batches.next()
                        self.names_queue.put(name_batch)
                    except StopIteration:
                        pass
                # The data_queue is empty: the epoch is over or we
                # consumed the data too fast
                except Queue.Empty:
                    if not self.names_queue.unfinished_tasks:
                        # END OF EPOCH - The data_queue is empty, i.e.
                        # the name_batches is empty: refill both
                        self._fill_names_batches(self.shuffle_at_each_epoch)
                        self._init_names_queue()  # Fill the names queue
                        if not self.infinite_iterator:
                            raise StopIteration
            else:
                # NO THREADS
                try:
                    name_batch = self.names_batches.next()
                    data_batch = self.fetch_from_dataset(name_batch)
                    done = True
                except StopIteration:
                    # END OF EPOCH - The name_batches is empty: refill it
                    self._fill_names_batches(self.shuffle_at_each_epoch)
                    if not self.infinite_iterator:
                        raise
                    # else, loop to the next image
                except IOError as e:
                    if self.raise_IOErrors:
                        raise
                    else:
                        print('WARNING: Image corrupted or missing!')
                        print(e)

        assert(data_batch is not None)
        return data_batch

    def fetch_from_dataset(self, batch_to_load):
        """
        Return *batches* of 5D sequences/clips or 4D images.

        `batch_to_load` contains the indices of the first frame/image of
        each element of the batch.
        `load_sequence` should return a numpy array of 2 or more
        elements, the first of which 4-dimensional (frame, 0, 1, c)
        or (frame, c, 0, 1) containing the data and the second 3D or 4D
        containing the label.
        """
        batch_ret = {}

        # Create batches
        for el in batch_to_load:

            if el is None:
                continue

            # Load sequence, format is (s, 0, 1, c)
            ret = self.load_sequence(el)
            assert all(el in ret.keys()
                       for el in ('data', 'labels', 'filenames', 'subset')), (
                    'Keys: {}'.format(ret.keys()))
            assert all(isinstance(el, np.ndarray)
                       for el in (ret['data'], ret['labels']))
            raw_data = ret['data'].copy()
            seq_x, seq_y = ret['data'], ret['labels']

            # Per-image normalization
            if self.remove_per_img_mean:
                seq_x -= seq_x.mean(axis=tuple(range(seq_x.ndim - 1)),
                                    keepdims=True)
            if self.divide_by_per_img_std:
                seq_x /= seq_x.std(axis=tuple(range(seq_x.ndim - 1)),
                                   keepdims=True)
            # Dataset statistics normalization
            if self.remove_mean:
                seq_x -= getattr(self, 'mean', 0)
            if self.divide_by_std:
                seq_x /= getattr(self, 'std', 1)

            # Make sure data is in 4D
            if seq_x.ndim == 3:
                seq_x = seq_x[np.newaxis, ...]
                raw_data = raw_data[np.newaxis, ...]
            assert seq_x.ndim == 4
            # and labels in 3D
            if self.set_has_GT:
                if seq_y.ndim == 2:
                    seq_y = seq_y[np.newaxis, ...]
                assert seq_y.ndim == 3

            # Perform data augmentation, if needed
            seq_x, seq_y = random_transform(
                seq_x, seq_y,
                nclasses=self.nclasses,
                void_label=self.void_labels,
                **self.data_augm_kwargs)

            if self.set_has_GT and self._void_labels != []:
                # Map all void classes to non_void_nclasses and shift the other
                # values accordingly, so that the valid values are between 0
                # and non_void_nclasses-1 and the void_classes are all equal to
                # non_void_nclasses.
                void_l = self._void_labels
                void_l.sort(reverse=True)
                mapping = self._mapping

                # Apply the mapping
                tmp_class = (-1 if not hasattr(self, 'GTclasses') else
                             max(self.GTclasses) + 1)
                seq_y[seq_y == self.non_void_nclasses] = tmp_class
                for i in sorted(mapping.keys()):
                    if i == self.non_void_nclasses:
                        continue
                    seq_y[seq_y == i] = mapping[i]
                try:
                    seq_y[seq_y == tmp_class] = mapping[self.non_void_nclasses]
                except KeyError:
                    # none of the original classes was self.non_void_nclasses
                    pass

            # Transform targets seq_y to one hot code if return_one_hot
            # is True
            if self.set_has_GT and self.return_one_hot:
                nc = (self.non_void_nclasses if self._void_labels == [] else
                      self.non_void_nclasses + 1)
                sh = seq_y.shape
                seq_y = seq_y.flatten()
                seq_y_hot = np.zeros((seq_y.shape[0], nc),
                                     dtype='int32')
                seq_y = seq_y.astype('int32')
                seq_y_hot[range(seq_y.shape[0]), seq_y] = 1
                seq_y_hot = seq_y_hot.reshape(sh + (nc,))
                seq_y = seq_y_hot

            # Dimshuffle if return_01c is False
            if not self.return_01c:
                # s,0,1,c --> s,c,0,1
                seq_x = seq_x.transpose([0, 3, 1, 2])
                if self.set_has_GT and self.return_one_hot:
                    seq_y = seq_y.transpose([0, 3, 1, 2])
                raw_data = raw_data.transpose([0, 3, 1, 2])

            # Return 4D images
            if not self.return_sequence:
                seq_x = seq_x[0, ...]
                if self.set_has_GT:
                    seq_y = seq_y[0, ...]
                raw_data = raw_data[0, ...]

            if self.return_0_255:
                seq_x = (seq_x * 255).astype('uint8')
            ret['data'], ret['labels'] = seq_x, seq_y
            ret['raw_data'] = raw_data
            # Append the data of this batch to the minibatch array
            for k, v in ret.iteritems():
                batch_ret.setdefault(k, []).append(v)

        for k, v in batch_ret.iteritems():
            try:
                batch_ret[k] = np.array(v)
            except ValueError:
                # Variable shape: cannot wrap with a numpy array
                pass
        if self.seq_length > 0 and self.return_middle_frame_only:
            batch_ret['labels'] = batch_ret['labels'][:, self.seq_length//2]
        if self.return_list:
            return [batch_ret['data'], batch_ret['labels']]
        else:
            return batch_ret

    def reset(self, shuffle, reload_sequences_from_dataset=True):
        '''Reset the dataset loader

        Resets the dataset loader according to the current parameters.
        If the parameters changed since initialization, by resetting the
        dataset they will be taken into account.

        Note that `reset` stops all the fetcher threads and makes sure the
        queues are emptied before adding new elements to them. This
        can introduce a slowdown in the fetching process. For this
        reason `shuffle` should be used to reshuffle the data when the
        parameters of the dataset have not been modified.
        '''
        if reload_sequences_from_dataset:
            # Get all the sequences of names from the dataset, with the
            # desired overlap
            self._fill_names_sequences()

        # Select the sequences we want, according to the parameters
        # Sets self.nsamples, self.nbatches and self.names_batches.
        # Self.names_batches is an iterator over the batches of names.
        self._fill_names_batches(shuffle)

        # Reset the queues
        if self.use_threads:
            # Empty names_queue
            with self.names_queue.mutex:
                done = False
                while not done:
                    try:
                        self.names_queue.get(False)
                        self.names_queue.task_done()
                    except Queue.Empty:
                        done = True
            # Wait for the fetchers to be done
            while not self.names_queue.unfinished_tasks:
                sleep(self._wait_time)
            # Empty the data_queue
            self.data_queue.queue.clear()
            self.data_queue.all_tasks_done.notify_all()
            self.data_queue.unfinished_tasks = 0

            # Refill the names queue
            self._init_names_queue()

    def shuffle(self):
        '''Shuffles the sequences and creates new batches, according to
        the initial parameters. To account for changes in the parameters
        of the dataset use `reset`.'''
        self._fill_names_batches(True)

    def finish(self):
        # Stop fetchers
        for _ in self.data_fetchers:
            self.names_queue.put(self.sentinel)
        while any([df() is not None and df().isAlive()
                   for df in self.data_fetchers]):
            sleep(self._wait_time)
        # Kill threads
        for data_fetcher in self.data_fetchers:
            if data_fetcher() is not None:
                data_fetcher().join()

    @classproperty
    def __config_parser__(self):
        config_parser = ConfigParser.ConfigParser()
        config_path = os.path.join(dataset_loaders.__path__[0], 'config.ini')
        if not os.path.isfile(config_path):
            raise RuntimeError('The config.ini is missing. Make sure to '
                               'create one in %s according to the '
                               'config.ini.example in the same '
                               'path.' % config_path)
        config_parser.read(config_path)
        return config_parser

    @classproperty
    def path(self):
        config_parser = self.__config_parser__
        return os.path.join(
            config_parser.get('general', 'datasets_local_path'),
            getattr(self, 'path_name', self.name))

    @classproperty
    def shared_path(self):
        config_parser = self.__config_parser__
        return config_parser.get(self.name, 'shared_path')

    @classproperty
    def nclasses(self):
        '''The number of classes in the output mask.'''
        return (self.non_void_nclasses + 1 if hasattr(self, '_void_labels') and
                self._void_labels != [] else self.non_void_nclasses)

    @classproperty
    def void_labels(self):
        '''Returns the void label(s)

        If the dataset has void labels, returns self.non_void_nclasses,
        i.e. the label to which all the void labels are mapped. Else,
        returns an empty list.'''
        return ([self.non_void_nclasses] if hasattr(self, '_void_labels') and
                self._void_labels != [] else [])

    @classproperty
    def _mapping(self):
        if hasattr(self, 'GTclasses'):
            mapping = {cl: i for i, cl in enumerate(
                sorted(set(self.GTclasses) - set(self._void_labels)))}
            for l in self._void_labels:
                mapping[l] = self.non_void_nclasses
        else:
            mapping = {}
            delta = 0
            # Prepare the mapping
            for i in range(self.non_void_nclasses + len(self._void_labels)):
                if i in self._void_labels:
                    mapping[i] = self.non_void_nclasses
                    delta += 1
                else:
                    mapping[i] = i - delta
        return mapping

    @classproperty
    def _inv_mapping(self):
        mapping = self._mapping
        return {v: k for k, v in mapping.items()}

    @classproperty
    def cmap(self):
        cmap = getattr(self, '_cmap', {})
        assert isinstance(cmap, dict)
        inv_mapping = self._inv_mapping
        cmap = np.array([cmap[inv_mapping[k]] for k in
                         sorted(inv_mapping.keys())])
        if cmap.max() > 1:
            # assume labels are in [0, 255]
            cmap = cmap / 255.  # not inplace or rounded to int
        return cmap

    @classproperty
    def mask_labels(self):
        mask_labels = getattr(self, '_mask_labels', {})
        assert isinstance(mask_labels, dict)
        if mask_labels == {}:
            return []
        inv_mapping = self._inv_mapping
        return np.array([mask_labels[inv_mapping[k]] for k in
                         sorted(inv_mapping.keys())])


def threaded_fetch(weakself):
    """
    Fill the data_queue.

    Whenever there are names in the names queue, it will read them,
    fetch the corresponding data and fill the data_queue.

    Note that in case of errors, it will put the exception object in the
    data_queue.
    """
    while True:
        sleep(0.1)  # Allow the gc to delete the main object if needed
        self = weakself()
        if self is None:
            break
        try:
            # Grabs names from queue
            batch_to_load = self.names_queue.get(False)

            if batch_to_load is self.sentinel:
                self.names_queue.task_done()
                del(self)
                break

            # Load the data
            minibatch_data = self.fetch_from_dataset(batch_to_load)

            # Place it in data_queue
            self.data_queue.put(minibatch_data)

            # Signal to the names queue that the job is done
            self.names_queue.task_done()
        except Queue.Empty:
            # names_queue is empty --> loop again
            pass
        except:
            # If any uncaught exception, pass it along and move on
            self.data_queue.put(sys.exc_info())
            self.names_queue.task_done()
        finally:
            del(self)
