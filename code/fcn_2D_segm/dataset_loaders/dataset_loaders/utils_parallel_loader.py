from itertools import izip, izip_longest
import os
import re


def get_video_size(path):
    """
    Return the lengths of each video.

    :path: path to find the data_size.txt file
    """
    f = open(os.path.join(path, "data_size.txt"), "r")
    video_size = []
    for line in f:
        video_size.append(int(line))

    return len(video_size), video_size


def get_frame_size(path, video_index, extension="tiff"):
    """
    Find height and width of frames from one video.

    :path: path of the dataset
    :video_index: index of the video
    """
    from skimage import io
    im_path = os.path.join(path, 'Original')
    if extension == "tiff":
        filename = str(video_index) + "_0.tiff"
    if extension == "jpg":
        filename = str(video_index) + "_0.jpg"
    img = io.imread(os.path.join(im_path, filename))

    return img.shape[0:2]


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split('(\d+)', text)]


class ClassPropertyDescriptor(object):

    def __init__(self, fget, fset=None):
        self.fget = fget
        self.fset = fset

    def __get__(self, obj, klass=None):
        if klass is None:
            klass = type(obj)
        return self.fget.__get__(obj, klass)()

    def __set__(self, obj, value):
        if not self.fset:
            raise AttributeError("can't set attribute")
        type_ = type(obj)
        return self.fset.__get__(obj, type_)(value)

    def setter(self, func):
        if not isinstance(func, (classmethod, staticmethod)):
            func = classmethod(func)
        self.fset = func
        return self


def classproperty(func):
    if not isinstance(func, (classmethod, staticmethod)):
        func = classmethod(func)

    return ClassPropertyDescriptor(func)


def grouper(iterable, n, fillvalue=None):
    '''grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx'''
    args = [iter(iterable)] * n
    return izip_longest(fillvalue=fillvalue, *args)


def overlap_grouper(iterable, n, prefix=None):
    '''overlap_grouper('AABCDD', 3, 'a') -->
        (('a', 'A'), ('a', 'A'), ('a', 'B')),
        (('a', 'A'), ('a', 'B'), ('a', 'C')),
        (('a', 'B'), ('a', 'C'), ('a', 'C'))'''
    if prefix:
        args = [zip([prefix] * len(iterable), iterable[el:])
                for el in range(n)]
        return izip(*args)
    else:
        args = [iter(iterable)] * n
        return izip(*args)
