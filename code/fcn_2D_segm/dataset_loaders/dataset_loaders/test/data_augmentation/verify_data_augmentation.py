import matplotlib.pyplot as plt
import scipy

from dataset_loaders.data_augmentation import random_transform


if __name__ == '__main__':
    face = scipy.misc.face()
    face = face[None, ...]  # b01c
    face = face / 255.

    # Show
    def show(img, title=''):
        plt.imshow(img[0])
        plt.title(title)
        plt.show()

    if False:
        show(face, 'face')
        # Rotation
        x, _ = random_transform(face, None,
                                rotation_range=150.,
                                fill_mode='constant',
                                chan_idx=3,
                                rows_idx=1,
                                cols_idx=2,
                                void_label=0)
        show(x, 'rotation')

        # Width shift
        x, _ = random_transform(face, None,
                                width_shift_range=0.3,
                                fill_mode='constant',
                                chan_idx=3,
                                rows_idx=1,
                                cols_idx=2,
                                void_label=0)
        show(x, 'width shift')

        # Height shift
        x, _ = random_transform(face, None,
                                height_shift_range=0.3,
                                fill_mode='constant',
                                chan_idx=3,
                                rows_idx=1,
                                cols_idx=2,
                                void_label=0)
        show(x, 'height shift')

        # Shear
        x, _ = random_transform(face, None,
                                shear_range=0.8,
                                fill_mode='constant',
                                chan_idx=3,
                                rows_idx=1,
                                cols_idx=2,
                                void_label=0)
        show(x, 'shear')

        # Zoom
        x, _ = random_transform(face, None,
                                zoom_range=(0.2, 0.4),
                                fill_mode='constant',
                                chan_idx=3,
                                rows_idx=1,
                                cols_idx=2,
                                void_label=0)
        show(x, 'zoom')

        # Chan shift
        x, _ = random_transform(face, None,
                                channel_shift_range=0.2,
                                chan_idx=3,
                                rows_idx=1,
                                cols_idx=2,
                                void_label=0)
        show(x, 'chan shift')

        # Horiz flip
        x, _ = random_transform(face, None,
                                horizontal_flip=1.,  # probability
                                chan_idx=3,
                                rows_idx=1,
                                cols_idx=2,
                                void_label=0)
        show(x, 'horiz flip')

        # Vert flip
        x, _ = random_transform(face, None,
                                vertical_flip=1.,  # probability
                                chan_idx=3,
                                rows_idx=1,
                                cols_idx=2,
                                void_label=0)
        show(x, 'vert flip')

        # Crop
        x, _ = random_transform(face, None,
                                crop_size=(100, 100),
                                chan_idx=3,
                                rows_idx=1,
                                cols_idx=2,
                                void_label=0)
        show(x, 'crop')

        # Gamma
        x, _ = random_transform(face, None,
                                gamma=0.5,
                                gain=2.,
                                chan_idx=3,
                                rows_idx=1,
                                cols_idx=2,
                                void_label=0)
        show(x, 'gamma')

    # Spline warp
    x, _ = random_transform(face, None,
                            spline_warp=True,
                            warp_sigma=8.5,
                            warp_grid_size=5,
                            chan_idx=3,
                            rows_idx=1,
                            cols_idx=2,
                            void_label=0)
    show(x, 'spline warp')
