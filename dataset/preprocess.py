import cv2
import numpy as np
from scipy.misc import imread

def load_image(path):
    """
    Load an image

    # Params
    - path : the path of the image file

    # Returns
    The image as a numpy array of ints of hape (height, width)
    """
    return imread(path)

def preprocess(image, target_size=(256, 256), augmentation=True, mask=None,
               zero_center=False, scale=1., dim_ordering='th', 
               to_bgr=False, flip=False, shift_x=0, shift_y=0, rot_range=0):
    """
    Preprocess an image, possibly with random augmentations and
    a mask with the same augmentations

    # Params
    - image : the image to preprocess
    - target_size : tuple (height, width) to resize the image to
    - augmentation : Whether to augment the image
    - mask : if not None, perform same flips, shifts etc. to this mask.
             the mask will be resized to target_size
    - zero_center : zero center the image (naively)
    - scale : multiply each pixel value in the image by this value
    - dim_ordering : if 'th', transpose image to (channel, height, width)
    - to_bgr : conver the image to BGR colorspace
    - flip : if True 50% chance of flipping the image horizontally
    - shift_x : randomly shift the image by this amount 
                in pixels horizontally [-shift_x, shift_x]
    - shift_y : randomly shift the image by this amount
                in pixels vertically [-shift_x, shift_x]
    - rot_range : rotate the image uniformly by 
                  [-rot_range, rot_range] degrees

    # Returns
    - preprocessed image
    - If specified: mask
    """
    cv2_imsize = (target_size[1], target_size[0])
    image = cv2.resize(image, cv2_imsize, interpolation=cv2.INTER_LINEAR)
    if to_bgr:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if mask is not None:
        mask = cv2.resize(mask, cv2_imsize, interpolation=cv2.INTER_NEAREST)

    if augmentation:
        # flip
        if flip and np.random.randint(2) == 1:
            image = np.fliplr(image)
            if mask is not None:
                mask = np.fliplr(mask)
        
        # translate
        shift_x = np.random.randint(-shift_x, shift_x+1)
        shift_y = np.random.randint(-shift_y, shift_y+1)
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        image = cv2.warpAffine(image, M, cv2_imsize)
        if mask is not None:
            mask = cv2.warpAffine(mask, M, cv2_imsize)

        # rotate
        rot = np.random.uniform(-rot_range, rot_range)
        # rotate wrt center
        M = cv2.getRotationMatrix2D((cv2_imsize[0]/2, cv2_imsize[1]/2), rot, 1)
        image = cv2.warpAffine(image, M, cv2_imsize)
        if mask is not None:
            mask = cv2.warpAffine(mask, M, cv2_imsize)
    
    if zero_center:
        image = image - 127 # naive zero-center
    image = image.astype(np.float32) * scale

    if dim_ordering == 'th':
        image = image.transpose(2, 0, 1)
        if mask is not None and mask.ndim == 3:
            mask = mask.transpose(2, 0, 1)

    if mask is not None:
        return image, mask

    return image
