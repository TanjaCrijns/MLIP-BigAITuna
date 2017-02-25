import os

import numpy as np
from keras.utils.np_utils import to_categorical as onehot

from preprocess import *
from dataset import labels

def get_data(data_df, data_folder, batch_size=32, shuffle=True, 
             augmentation=True, img_size=(256, 256), **kwargs):
    """
    Generator to train a model on images.

    # Params
    - data_df : DataFrame of filename and label for each image
    - data_folder : folder where data resides. Should have structure 
                    `data_folder/label/img_name`
    - batch_size : number of images per batch
    - shuffle : present images in random order (each epoch)
    - augmentation : perform data augmentation
    - img_size : sample patches of this size from the image and mask
    - kwargs : passed to the preprocess function

    # Returns
    - batch of images (batch_size, 3, img_size[0], img_size[1])
    """
    n_classes = len(labels)
    
    while True:
        data = zip(data_df.filename.values, data_df.label.values)
        n = len(data)
        if shuffle:
            data = np.random.permutation(data)
        data = list(data)

        # Double to allow for larger batch sizes
        data += data
        i = 0
        while i < n:
            img_batch = np.zeros((batch_size, 3) + img_size, dtype=np.float32)
            label_batch = np.zeros((batch_size, n_classes), dtype=np.uint8)
            for j in range(batch_size):
                img_name, label = data[i + j]
                img_path = os.path.join(data_folder, label, img_name)
                img = load_image(img_path)
                
                img = preprocess(img, target_size=img_size, 
                                augmentation=augmentation, 
                                zero_center=True, scale=1./255.,
                                **kwargs)
                
                img_batch[j] = img
                label_batch[j] = onehot(labels.index(label), 8)
                i += 1
            yield img_batch, label_batch

def get_data_with_masks(data_df, bboxes, data_folder, batch_size=1,
                        shuffle=True, augmentation=True, img_size=(256, 256),
                        **kwargs):
    """
    Generator to train a model on images with both segmentation and
    whole image label.

    # Params
    - data_df : DataFrame of filename and label for each image
    - bboxes : dictionary of 
               img_name -> [(x, y, width, height, class), ...]
    - data_folder : folder where data resides. Should have structure 
                    `data_folder/label/img_name`
    - batch_size : number of images per batch
    - shuffle : present images in random order (each epoch)
    - augmentation : perform data augmentation
    - img_size : sample patches of this size from the image and mask
    - kwargs : passed to the preprocess function

    # Returns
    - batch of images (batch_size, 3, img_size[0], img_size[1])
    - a list of: 
            1. labels (batch_size, n_classes)
            2. masks (batch_size, 1, img_size[0], img_size[1])
    """
    n_classes = len(labels)
    
    while True:
        data = zip(data_df.filename.values, data_df.label.values)
        n = len(data)
        if shuffle:
            data = np.random.permutation(data)
        data = list(data)

        # Double to allow for larger batch sizes
        data += data
        i = 0
        while i < n:
            img_batch = np.zeros((batch_size, 3) + img_size, dtype=np.float32)
            mask_batch = np.zeros((batch_size, 1) + img_size, dtype=np.uint8)
            label_batch = np.zeros((batch_size, n_classes), dtype=np.uint8)
            for j in range(batch_size):
                img_name, label = data[i + j]
                img_path = os.path.join(data_folder, label, img_name)
                img = load_image(img_path)
                mask = np.zeros(img.shape[:2], dtype=np.uint8)
                
                # Create the mask
                for x, y, width, height, _ in bboxes[img_name]:
                    mask[y:y+height, x:x+width] = 1
                
                img, mask = preprocess(img, target_size=img_size, 
                                       augmentation=augmentation, mask=mask, 
                                       zero_center=True, scale=1./255.,
                                       **kwargs)
                
                img_batch[j] = img
                mask_batch[j] = np.expand_dims(mask, 0)
                label_batch[j] = onehot(labels.index(label), 8)
                i += 1
            yield img_batch, [label_batch, mask_batch]