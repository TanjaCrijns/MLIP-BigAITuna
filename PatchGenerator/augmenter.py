import scipy.misc as spm
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage.interpolation as sni
import scipy.ndimage.filters as snf


class Augmenter(object):

    def __init__(self, name):
        super(Augmenter, self).__init__()
        self.keyword = name

    def augment(self, patch, label):
        pass

    def randomize(self):
        pass


class PatchFlip(Augmenter):

    def __init__(self, vertical_flip, horizontal_flip):
        super(PatchFlip, self).__init__('flip')
        self.flip_set = []
        if vertical_flip:
            self.flip_set.append(0)
        if horizontal_flip:
            self.flip_set.append(1)
        if vertical_flip and horizontal_flip:
            self.flip_set.append(2)
        self.__flip_direction = None

    def augment(self, patch, label=None):
        augments = []
        if self.__flip_direction % 2 != 1:  # Add ud flip if 0 or 2
            augments.append(np.flipud)
        if self.__flip_direction > 0:  # Add lr flip if 1 or 2
            augments.append(np.fliplr)
        t_patch = patch.copy().transpose(1, 2, 0)
        t_label = None
        if label is not None:
            t_label = label.copy().transpose(1, 2, 0)
        for f in augments:
            t_patch = f(t_patch)
            if label is not None:
                t_label = f(t_label)
        t_patch = t_patch.transpose(2, 0, 1)
        if label is not None:
            t_label.transpose(2, 0, 1)
        return t_patch, t_label

    def randomize(self):
        self.__flip_direction = np.random.choice(self.flip_set)


class PatchRotate90(Augmenter):

    def __init__(self, k_list):
        super(PatchRotate90, self).__init__('rotate90')
        # K is the parameter that regulates the amount of rotations done.
        self.k_list = k_list
        self.__k = None

    def augment(self, patch, label=None):
        t_patch = np.rot90(patch.transpose(1, 2, 0),
                           self.__k).transpose(2, 0, 1)
        t_label = None
        if label is not None:
            t_label = np.rot90(label.transpose(1, 2, 0),
                               self.__k).transpose(2, 0, 1)
        return t_patch, t_label

    def randomize(self):
        self.__k = np.random.choice(self.k_list)


class PatchRotate(Augmenter):

    def __init__(self, angle_range):
        # Angle should be in degrees, possible range is:
        # (-angle_range, +angle_range).
        super(PatchRotate, self).__init__('rotate')
        self.angle_range = angle_range
        self.__rotation_angle = None

    def augment(self, patch, label=None):
        t_patch = sni.rotate(patch, angle=self.__rotation_angle, axes=(1, 2),
                             mode='constant', cval=1.0)
        t_label = None
        if label is not None:
            t_label = sni.rotate(patch, angle=self.__rotation_angle,
                                 axes=(1, 2), mode='constant', cval=1.0)
        return t_patch, t_label

    def randomize(self):
        self.__rotation_angle = np.random.uniform(-self.angle_range,
                                                  self.angle_range)


class PatchBlur(Augmenter):

    def __init__(self, sigma_range):
        # Sigma_range should be a tuple (a, b) to indicate the range
        super(PatchBlur, self).__init__('blur')
        self.sigma_range = sigma_range
        self.__sigma = None

    def augment(self, patch, label=None):
        t_patch = snf.gaussian_filter(patch, (0.0,
                                              self.__sigma,
                                              self.__sigma))
        return t_patch, label

    def randomize(self):
        self.__sigma = np.random.uniform(self.sigma_range[0],
                                         self.sigma_range[1])

class PatchGamma(Augmenter):

    def __init__(self, gamma_interval):
        super(PatchGamma, self).__init__('gamma')
        self.gamma_interval = gamma_interval
        self.__gamma = None

    def augment(self, patch, label=None):
        t_patch = np.rint(np.power(patch.astype(np.float32) / 255.0,
                          self.__gamma) * 255.0).astype(np.uint8)
        return t_patch, label

    def randomize(self):
        self.__gamma = np.random.uniform(self.gamma_interval[0],
                                         self.gamma_interval[1])

if __name__ == '__main__':
    example_image = spm.imread('../data/train/YFT/img_03596.jpg')
    example_image = example_image.transpose(2, 0, 1)
    augmenter = PatchGamma((0.1, 0.9))
    augmenter.randomize()
    example, _ = augmenter.augment(example_image)
    example = example.transpose(1, 2, 0)
    plt.imshow(example)
    plt.show()
