import os

import numpy as np
from PIL import Image

from dataset_loaders.parallel_loader import ThreadedDataset

floatX = 'float32'


class ChangeDetectionDataset(ThreadedDataset):
    '''The Change Detection 2014 dataset

    The Change Detection dataset 2014 [1]_ consists of 11 video
    categories with 4 to 6 video sequences in each category where the
    goal is to identify changing or moving areas in the field of view of
    a camera.

    The videos are divided into different categories according to the
    main difficulty of each video, e.g., sudden illumination changes,
    environmental conditions (night, rain, snow, air turbulence),
    background/camera motion, shadows, and camouflage effects
    (photometric similarity of object and background). The categories
    are provided in `self.categories`. Specific subsets of categories
    and videos can be loaded specifying the `which_category` and
    `which_video` arguments.

    Each video has associated a temporalROI and a ROI. The temporalROI
    determines which frames are to be used for training and test. The
    ROI defines the area of the frame that we are interested in. This
    loader processes the temporalROI to split the data transparently.

    The dataset should be downloaded from [1]_ into the `shared_path`
    (that should be specified in the config.ini according to the
    instructions in ../README.md).

    Parameters
    ----------
    which_set: string
        A string in ['train', 'val', 'test'], corresponding to the set
        to be returned.
    which_category: list of strings
        A list of the categories to be loaded.
    which_video: list of strings
        A list of the videos to be loaded.
    split: float
        The percentage of the training data to be used for validation.
        The first `split`\% of the training set will be used for
        training and the rest for validation. Default: 0.75.
    verbose: bool
        If True debug information will be printed when the dataset is
        loaded.

     References
    ----------
    .. [1] http://changedetection.net/
    '''
    name = 'change_detection'
    non_void_nclasses = 4
    _void_labels = [85]

    mean = [0.45483398, 0.4387207, 0.40405273]
    std = [0.04758175, 0.04148954, 0.05489637]
    GTclasses = [0, 50, 85, 170, 255]
    categories = ['badWeather', 'baseline', 'cameraJitter',
                  'dynamicBackground', 'intermittentObjectMotion',
                  'lowFramerate', 'nightVideos', 'PTZ', 'shadow', 'thermal',
                  'turbulence']
    # with void [0.81749293, 0.0053444, 0.55085129, 0.03961262, 0.45756284]
    class_freqs = [0.81749293, 0.0053444, 0.03961262, 0.45756284]

    # static, shadow, ground, solid (buildings, etc), porous, cars, humans,
    # vert mix, main mix
    _cmap = {
        0: (0, 0, 0),           # static
        50: (255, 0, 0),        # shadow (red)
        170: (0, 255, 0),       # unknown (green)
        255: (255, 255, 255),   # moving (white)
        85: (127, 127, 127)}    # non-roi (grey)
    _mask_labels = {0: 'static', 50: 'shadow', 170: 'unknown', 255: 'moving',
                    85: 'non-roi'}

    _filenames = None

    @property
    def filenames(self):
        if self._filenames is None:
            inspect_dataset_properties = False  # debugging purpose
            self._filenames = {}
            ROIS = {}
            ROIS2 = {}
            tempROIS = {}
            cat_videos = {}
            for root, dd, ff in os.walk(self.path):
                if ff == [] or 'README' in ff:
                    # Root or category dir
                    dd.sort(key=str.lower)
                elif 'ROI.jpg' in ff:
                    # Video dir
                    category, video = root.split('/')[-2:]
                    cat_videos.setdefault(category, []).append(video)
                    ROI = np.array(Image.open(os.path.join(root, 'ROI.jpg')))
                    ROI2 = np.array(Image.open(os.path.join(root,
                                                            'ROI.bmp.tif')))
                    ROIS[video] = ROI
                    ROIS2[video] = ROI2
                    tempROIS[video] = open(os.path.join(
                        root, 'temporalROI.txt'), 'r').readline().split(' ')
                else:
                    # Images or GT dir
                    category, video, kind = root.split('/')[-3:]

                    if (category not in self.which_category or
                            video not in self.which_video):
                        continue

                    ff.sort(key=str.lower)
                    ff = [fname for fname in ff if 'Thumbs.db' not in fname]

                    if not inspect_dataset_properties:
                        # 1-indexed, inclusive
                        s, e = [int(el) - 1 for el in tempROIS[video]]
                        if self.which_set == 'test':
                            # anything out of tempROI
                            ff = ff[0:s] + ff[e+1:]
                        elif self.which_set == 'train':
                            d = int((e+1-s)*(1 - self.split))  # valid_delta
                            ff = ff[s+d:e+1]
                        else:
                            d = int((e+1-s)*(1 - self.split))  # valid delta
                            ff = ff[s:s+d]

                    if kind == 'input':
                        if self.verbose:
                            print('Loading {}..'.format(root[len(self.path):-6]))

                        self._filenames.setdefault(video, {}).update(
                            {'category': category,
                             'root': root[:-6],  # remove '/input'
                             'images': ff,
                             'ROI': ROIS[video],
                             'tempROI': tempROIS[video]})
                    else:
                        self._filenames.setdefault(video, {}).update(
                            {'GTs': ff})

            # Dataset properties:
            if inspect_dataset_properties:
                kk = self._filenames.keys()
                for k in kk:
                    tempROI = self._filenames[k]['tempROI']
                    # temporalROI is either at the beginning or at the end of
                    # the sequence
                    assert (int(tempROI[0]) == 001 or
                            int(tempROI[1]) == len(
                                self._filenames[k]['images'])), k
                    # First gt is gt000001.png
                    assert self._filenames[k]['GTs'][0] == 'gt000001.png', k
                    # First im is in000001.jpg
                    assert self._filenames[k]['images'][0] == 'in000001.jpg', k
                    # GT outside of tempROI is always 85 (void)
                    for i, f in enumerate(self._filenames[k]['images']):
                        if i < tempROI[0] or i > tempROI[1]:
                            continue  # consider only frames in tempROI
                        path = self._filenames[k]['root'] + '/' + f
                        gt = np.array(Image.open(path))
                        labels = np.unique(gt)
                        if len(labels) != 1:
                            print('k {} i {} labels {}'.format(k, i, labels))
                        if labels[0] != 85:
                            print('Non 85: k {} i {}'.format(k, i))

        return self._filenames

    def __init__(self,
                 which_set='train',
                 split=.75,
                 which_category=('badWeather', 'baseline',
                                 'cameraJitter',
                                 'dynamicBackground',
                                 'intermittentObjectMotion', 'lowFramerate',
                                 'nightVideos', 'PTZ', 'shadow', 'thermal',
                                 'turbulence'),
                 which_video=(
                     # badWeather
                     'blizzard', 'skating', 'snowFall', 'wetSnow',
                     # baseline
                     'highway', 'office', 'pedestrians', 'PETS2006',
                     # cameraJitter
                     'badminton', 'boulevard', 'sidewalk', 'traffic',
                     # dynamicBackground
                     'boats', 'canoe', 'fall', 'fountain01', 'fountain02',
                     'overpass',
                     # intermittentObjectMotion
                     'abandonedBox', 'parking', 'sofa', 'streetLight',
                     'tramstop', 'winterDriveway',
                     # lowFramerate
                     'port_0_17fps', 'tramCrossroad_1fps',
                     'tunnelExit_0_35fps', 'turnpike_0_5fps',
                     # nightVideos
                     'bridgeEntry', 'busyBoulvard', 'fluidHighway',
                     'streetCornerAtNight', 'tramStation', 'winterStreet',
                     # PTZ
                     'continuousPan', 'intermittentPan',
                     'twoPositionPTZCam', 'zoomInZoomOut',
                     # shadow
                     'backdoor', 'bungalows', 'busStation', 'copyMachine',
                     'cubicle', 'peopleInShade',
                     # thermal
                     'corridor', 'diningRoom', 'lakeSide', 'library', 'park',
                     # turbulence
                     'turbulence0', 'turbulence1', 'turbulence2',
                     'turbulence3'),
                 verbose=False,
                 *args, **kwargs):

        self.which_set = 'valid' if which_set == 'val' else which_set
        assert self.which_set in ['train', 'valid', 'test'], self.which_set
        self.split = split
        self.which_category = which_category
        self.which_video = which_video
        if self.which_set == 'test':
            self.set_has_GT = False
            print('No mask for the test set!!')
        self.verbose = verbose

        super(ChangeDetectionDataset, self).__init__(*args, **kwargs)

    def get_names(self):
        """Return a dict of names, per prefix/subset."""
        per_video_names = {}
        self.video_length = {}

        # cycle through the videos
        for video, data in self.filenames.iteritems():
            video_length = len(data['images'])
            self.video_length[video] = video_length
            # append a bunch of tuples (video_name, idx)
            per_video_names[video] = [idx for idx in range(video_length)]
        return per_video_names

    def load_sequence(self, sequence):
        """Load a sequence of images/frames

        Auxiliary function that loads a sequence of frames with
        the corresponding ground truth and their filenames.
        Returns a dict with the images in [0, 1], their corresponding
        labels, their subset (i.e. category, clip, prefix) and their
        filenames.
        """
        from skimage import io
        X = []
        Y = []
        F = []

        for video, idx in sequence:
            data = self.filenames[video]
            root = data['root']

            im, gt = data['images'][idx], data['GTs'][idx]
            img = io.imread(os.path.join(self.path, root, 'input', im))
            mask = io.imread(os.path.join(self.path, root, 'groundtruth', gt))

            img = img.astype(floatX) / 255.
            mask = mask.astype('int32')

            X.append(img)
            Y.append(mask)
            F.append(im)

        # test only has void. No point in considering the mask
        Y = [] if self.which_set == 'test' else Y

        ret = {}
        ret['data'] = np.array(X)
        ret['labels'] = np.array(Y)
        ret['subset'] = video
        ret['filenames'] = np.array(F)
        return ret


def test():
    train = ChangeDetectionDataset(
        which_set='train',
        batch_size=1,
        seq_per_subset=0,
        seq_length=0,
        shuffle_at_each_epoch=False,
        infinite_iterator=False,
        split=.75)
    valid = ChangeDetectionDataset(
        which_set='valid',
        batch_size=1,
        seq_per_subset=0,
        seq_length=0,
        shuffle_at_each_epoch=False,
        infinite_iterator=False,
        split=.75)
    test = ChangeDetectionDataset(
        which_set='test',
        batch_size=1,
        seq_per_subset=0,
        seq_length=0,
        shuffle_at_each_epoch=False,
        infinite_iterator=False,
        split=.75)
    data = {'train': train, 'valid': valid, 'test': test}

    train_nsamples = train.nsamples
    valid_nsamples = valid.nsamples
    test_nsamples = test.nsamples
    print("#samples: train %d, valid %d, test %d" % (train_nsamples,
                                                     valid_nsamples,
                                                     test_nsamples))

    for split in ['train', 'valid', 'test']:
        nbatches = data[split].nbatches
        for i, mb in enumerate(range(nbatches)):
            el = data[split].next()
            if el['data'].min() < 0:
                raise Exception('Image {} of {} is smaller than 0'.format(
                    el['filenames'], split, el['data'].min()))
            if el['data'].max() > 1:
                raise Exception('Image {} of {} is greater than 1'.format(
                    el['filenames'], split, el['data'].max()))
            if split is not 'test' and el['labels'].max() > 4:
                raise Exception('Mask {} of {} is greater than 4: {}'.format(
                    el['filenames'], split, el['labels'].max()))
            if split is not 'test' and np.unique(el['labels']).tolist() == [4]:
                # check if the images is actually all void
                filename = el['filenames'][0, 0, 0]
                f = filename[-10:-3]
                mask_f = filename[:-18] + 'groundtruth/gt' + f + 'png'
                un = np.unique(Image.open(mask_f))
                if un.tolist() != [85]:  # discard test
                    raise Exception('Image {} of {} is not test and is all '
                                    'void:{}. It should be {}'.format(
                                        el['filenames'], split,
                                        np.unique(el['labels']), un))
                # else:
                #     print('Image {} of {} is not test and is all void. '
                #           'Weird, but not an issue of the wrapper')
            if i % 1000 == 0:
                print('Sample {}/{} of {}'.format(i, data[split].nsamples,
                                                  split))
        print('Split {} done!'.format(split))


def run_tests():
    test()


if __name__ == '__main__':
    run_tests()
