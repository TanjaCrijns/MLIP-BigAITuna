import cv2
import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

def to_categorical(y, nb_classes=None):
    """Converts a class vector (integers) to binary class matrix.

    E.g. for use with categorical_crossentropy.

    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to nb_classes).
        nb_classes: total number of classes.

    # Returns
        A binary matrix representation of the input.
    """
    y = np.array(y, dtype='int').ravel()
    if not nb_classes:
        nb_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, nb_classes))
    categorical[np.arange(n), y] = 1
    return categorical

def load_image(path):
    """
    Load an image

    # Params
    - path : the path of the image file

    # Returns
    The image as a numpy array of ints of shape (height, width)
    """
    return cv2.imread(path, cv2.IMREAD_COLOR)[:, :, ::-1]

def preprocess(image, target_size=None, augmentation=True, mask=None,
               zero_center=False, scale=1., dim_ordering='th',
               to_bgr=False, flip=False, shift_x=0, shift_y=0, rot_range=0,
               elastic_trans=False, colorize=False):
    """
    Preprocess an image, possibly with random augmentations and
    a mask with the same augmentations

    # Params
    - image : the image to preprocess
    - target_size : tuple (height, width) to resize the image to,
                    if None the image is not resized.
    - augmentation : Whether to augment the image
    - mask : if not None, perform same flips, shifts etc. to this mask.
             the mask will be resized to target_size
    - zero_center : zero center the image (naively)
    - scale : multiply each pixel value in the image by this value
    - dim_ordering : if 'th', transpose image to (channel, height, width)
    - to_bgr : convert the image to BGR colorspace
    - flip : if True 50% chance of flipping the image horizontally
    - shift_x : randomly shift the image by this amount 
                in pixels horizontally [-shift_x, shift_x]
    - shift_y : randomly shift the image by this amount
                in pixels vertically [-shift_x, shift_x]
    - rot_range : rotate the image uniformly by 
                  [-rot_range, rot_range] degrees
    - elastic_transform : transform the image elastically
    - colorize : Apply histogram equalization with one of the 
                 precalculated histograms

    # Returns
    - preprocessed image
    - If specified: mask
    """
    image_size = image.shape
    cv2_imsize = (image_size[1], image_size[0])
    
    if target_size is not None:
        cv2_imsize = (target_size[1], target_size[0])
        image = cv2.resize(image, cv2_imsize, interpolation=cv2.INTER_LINEAR)
        
    if to_bgr:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
    if mask is not None:
        mask = cv2.resize(mask, cv2_imsize, interpolation=cv2.INTER_NEAREST)

    
    if augmentation:
        # histogram normalization
        if colorize:
            file_index = np.random.randint(5)
            files = ['img_07419.txt', 'img_00230.txt', 'img_01384.txt', 'img_02487.txt', 'img_00726.txt']
            target = np.loadtxt('../dataset/color normalization histograms/histogram_' + files[file_index])
            target = target/target.sum()
            target = target * cv2_imsize[1] * cv2_imsize[0] * 3
            
            image = histogram_colorization(target, image)
        
        
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
        
        # elastic transform
        if elastic_trans:
            image = elastic_transform(image)
        
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

def elastic_transform(image):
    """
    Based on: https://www.kaggle.com/bguberfain/ultrasound-nerve-segmentation/elastic-transform-for-data-augmentation

    Transform an image elastically as a form of data augmentation

    # Params
    - image : the image to transform

    # Returns
    - the transformed image
    """
        
    ran = np.random.randint(4)
    alpha = image.shape[1] * ran
    sigma = image.shape[1] * 0.2
    alpha_affine = image.shape[1] * 0.035
    random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]
    
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)
    
    blur_size = int(4*sigma) | 1
    
    #Blur at half resolution
    dx = cv2.GaussianBlur(image[::4,::4], ksize=(blur_size, blur_size), sigmaX=sigma)
    dy = cv2.GaussianBlur(image[::4,::4], ksize=(blur_size, blur_size), sigmaX=sigma)
    
    dx = cv2.resize(dx, dsize=(dx.shape[0]*4, dx.shape[1]*4)).transpose(1,0,2)
    dy = cv2.resize(dy, dsize=(dy.shape[0]*4, dy.shape[1]*4)).transpose(1,0,2)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)

def histogram_colorization(target_hist, input_img, n_bins=255):
    input_img = input_img/float(n_bins)
    
    normalized_img = np.zeros_like(input_img)
    for i in [0, 1, 2]:
        hist_input = np.histogram(input_img[:, :, i], bins=np.linspace(0., 1., n_bins+1))[0]
        LUT, _, _ = get_histogram_matching_lut(hist_input, np.round(target_hist[i]).astype(np.int))
        stain_lut = LUT[(input_img[:, :, i] * (n_bins-1)).astype(int)].astype(float) / float(n_bins)
        normalized_img[:, :, i] = stain_lut
        
        hist_output = np.histogram(normalized_img[:, :, i], bins=np.linspace(0., 1., n_bins+1))[0]
    
    # output is in range [0, 1], rescale to [0, 255]
    # this may not be desirable in all cases
    return normalized_img * n_bins

def get_histogram_matching_lut(h_input, h_template):
    ''' h_input: histogram to transfrom, h_template: reference'''
    
    if len(h_input) != len(h_template):
        print 'histograms length mismatch!'
        return False
    
    H_input = np.cumsum(h_input)
    H_template = np.cumsum(h_template)
    LUT = np.array([np.argmin(np.abs(i - H_template)) for i in H_input])

    return LUT,H_input,H_template
