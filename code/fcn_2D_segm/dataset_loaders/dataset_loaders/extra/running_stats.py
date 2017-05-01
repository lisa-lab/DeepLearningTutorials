import numpy
import tables


class RunningStats:
    """Computes running mean and standard deviation
    Adapted from:
        *
        <http://stackoverflow.com/questions/1174984/how-to-efficiently-\
calculate-a-running-standard-deviation>
        * <http://mathcentral.uregina.ca/QQ/database/QQ.09.02/carlos1.html>

    Example
    -------
        Given `dataset`, a list of images, this code would compute the
        per-pixel statistics::

            runner = RunningStats()
            for img in dataset:
                runner.push(img)
            print(runner.mean())
            print(runner.std())

    Example
    -------
        Given `dataset`, a list of masks, this code would compute the
        class frequency::

            runner = RunningStats(compute_class_freq=True, nclasses=10)
            for mask in dataset:
                runner.push(mask)
            print(runner.class_freqs())
    """

    def __init__(self, compute_class_freq=False, nclasses=None):
        ''' An object to collect running stats

            Parameters
            ----------
            compute_class_freq: bool
                If False, mean and std_dev will be computed, else class
                frequency will be computed. In the first case the
                expected input is the image while in the second is the
                mask
            nclasses = int
                The number of classes in the dataset. Only used if
                compute_class_freq is True (required in that case)
        '''
        self.n = 0.
        self.compute_class_freq = compute_class_freq
        self.nclasses = nclasses

        if compute_class_freq:
            if not nclasses:
                raise RuntimeError('To compute class frequencies, provide '
                                   'nclasses')
            self.class_counts = numpy.zeros(self.nclasses)
            self.class_tot_px = numpy.zeros(self.nclasses)

    def clear(self):
        self.n = 0.

    def push(self, x, per_dim=True):
        x = numpy.array(x).copy().astype('float16')
        # process input
        if per_dim:
            self.update_params(x)
        else:
            for el in x.flatten():
                self.update_params(el)

    def update_params(self, x):
        # class freq
        if self.compute_class_freq:
            cl_ids, cl_counts = numpy.unique(x, return_counts=True)
            tot_px = numpy.sum(cl_counts)
            for cl_id, cl_count in zip(cl_ids, cl_counts):
                self.class_counts[cl_id] += cl_count
                self.class_tot_px[cl_id] += tot_px
        else:
            self.n += 1
            # mean, std_dev
            if self.n == 1:
                self.m = x
                self.s = 0.
            else:
                prev_m = self.m.copy()
                self.m += (x - self.m) / self.n
                self.s += (x - prev_m) * (x - self.m)

    def mean(self):
        return self.m if self.n else 0.0

    def variance(self):
        return self.s / (self.n) if self.n else 0.0

    def std(self):
        return numpy.sqrt(self.variance())

    def class_freqs(self):
        return self.class_counts / self.class_tot_px


def test_running_stats():
    from numpy.testing import assert_almost_equal as almost_equal
    from numpy.random import randint
    arr = numpy.random.randn((30*8*12)).reshape((30, 8, 12))
    varsize_arr = []
    for el in arr:
        s = el.shape
        varsize_arr.append(el[:randint(s[0]-2)+2, :randint(s[1]-2)+2])

    # test per dimension statistics
    perdim_runner = RunningStats()
    for i, el in enumerate(arr, 1):
        perdim_runner.push(el)
        if i == 1:
            # arr[:i] has no axis 0
            continue
        almost_equal(arr[:i].mean(axis=0), perdim_runner.mean())
        almost_equal(arr[:i].std(axis=0), perdim_runner.std())

    # test single number statistics
    runner = RunningStats()
    for i, el in enumerate(varsize_arr, 1):
        runner.push(el, False)
        cum_arr = []
        for im in varsize_arr[:i]:
            cum_arr = numpy.concatenate([im.flatten(), cum_arr])
        almost_equal(numpy.array(cum_arr).mean(), runner.mean())
        almost_equal(numpy.array(cum_arr).std(), runner.std())


def preprocess(arr, shape):
    if type(arr) is tables.VLArray:
        newarr = VLArrayWrapper(arr, shape)
    if type(arr) is tables.EArray:
        newarr = EArrayWrapper(arr, shape)
    return newarr


class VLArrayWrapper(tables.VLArray):
    def __init__(self, tables_arr, shape):
        '''Inspired by
        http://code.activestate.com/recipes/389916-example-setattr-getattr-\
overloading/
        '''
        self.__tables_arr__ = tables_arr
        self.__arr_shape__ = shape
        self.__current_iter__ = 0

    def __getattr__(self, attr):
        # NOTE do not use hasattr, it goes into infinite recurrsion
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self.__tables_arr__, attr)

    def __setattr__(self, attr, val):
        """Maps attributes to values.
        Only if we are initialised
        """
        # this test allows attributes to be set in the __init__ method
        if '_EArrayWrapper__initialised' not in self.__dict__:
            return dict.__setattr__(self, attr, val)
        self.__tables_arr__.__setattr__(attr, val)

    def __hasattr__(self, attr):
        return self.__tables_arr__.hasAttr(attr)

    def __getitem__(self, index):
        arr = self.__tables_arr__[index]
        arr = (arr).astype('uint8')
        return arr.reshape(self.__arr_shape__[index]) \
            if self.__arr_shape__ else arr

    def __iter__(self):
        return self

    def next(self):
        arr = self.__tables_arr__.next()
        self.__current_iter__ += 1
        return (arr).astype('uint8').reshape(self.__arr_shape__(
                    self.__current_iter__ - 1))


class EArrayWrapper(tables.EArray):
    def __init__(self, tables_arr, shape):
        '''Inspired by
        http://code.activestate.com/recipes/389916-example-setattr-getattr-\
overloading/
        '''
        self.__tables_arr__ = tables_arr
        self.__arr_shape__ = shape
        self.__current_iter__ = 0

    def __getattr__(self, attr):
        # NOTE do not use hasattr, it goes into infinite recurrsion
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self.__tables_arr__, attr)

    def __setattr__(self, attr, val):
        """Maps attributes to values.
        Only if we are initialised
        """
        # this test allows attributes to be set in the __init__ method
        if '_EArrayWrapper__initialised' not in self.__dict__:
            return dict.__setattr__(self, attr, val)
        self.__tables_arr__.__setattr__(attr, val)

    def __hasattr__(self, attr):
        return self.__tables_arr__.hasAttr(attr)

    def __getitem__(self, index):
        arr = self.__tables_arr__[index]
        arr = (arr).astype('uint8')
        return arr.reshape(self.__arr_shape__[index]) \
            if self.__arr_shape__ else arr

    def __iter__(self):
        return self

    def next(self):
        arr = self.__tables_arr__.next()
        self.__current_iter__ += 1
        return (arr).astype('uint8').reshape(self.__arr_shape__(
                    self.__current_iter__ - 1))
