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
            mask_batch = np.zeros((batch_size, 2) + img_size, dtype=np.uint8)
            label_batch = np.zeros((batch_size, n_classes), dtype=np.uint8)
            for j in range(batch_size):
                img_name, label = data[i + j]
                img_path = os.path.join(data_folder, label, img_name)
                img = load_image(img_path)
                mask = np.zeros(img.shape[:2] + (2,), dtype=np.uint8)
                mask[:, :, 0] = 1
                
                # Create the mask
                for x, y, width, height, _ in bboxes[img_name]:
                    mask[y:y+height, x:x+width, 1] = 1
                    mask[y:y+height, x:x+width, 0] = 0
                
                # Only use bboxes with the same label
                # Other fish may be present on the image
                relevant_bboxes = [bbox for bbox in bboxes[img_name]
                                            if bbox[4] == label]
                if len(relevant_bboxes) != 0:
                    # Choose bbox to center on
                    bbox_idx = np.random.choice(len(relevant_bboxes))
                    x, y, width, height, _ = relevant_bboxes[bbox_idx]
                    # The fish should be fully on the patch
                    min_x = min(max(0, x + width - img_size[1]), img.shape[1] - img_size[1])
                    min_y = min(max(0, y + height - img_size[0]), img.shape[0] - img_size[0])
                    max_x = min(img.shape[1] - img_size[1], x)
                    max_y = min(img.shape[0] - img_size[0], y)
                    # temporary fix, not sure why this happens
                    # TODO
                    if min_x >= max_x:
                        min_x = 0
                        max_x = 1
                    if min_y >= max_y:
                        min_y = 0
                        max_y = 1
                    x_crop = np.random.randint(min_x, max_x)
                    y_crop = np.random.randint(min_y, max_y)
                else:
                    # No relevant bbox, pick a random patch
                    x_crop = np.random.randint(0, img.shape[1] - img_size[1])
                    y_crop = np.random.randint(0, img.shape[0] - img_size[0])

                img = img[y_crop:y_crop+img_size[0], x_crop:x_crop+img_size[1]]
                mask = mask[y_crop:y_crop+img_size[0], x_crop:x_crop+img_size[1]]

                img, mask = preprocess(img, target_size=img_size, 
                                       augmentation=augmentation, mask=mask, 
                                       zero_center=True, scale=1./255.,
                                       **kwargs)

                img_batch[j] = img
                mask_batch[j] = mask
                label_batch[j] = onehot(labels.index(label), 8)
                i += 1
            # yield img_batch, [label_batch, mask_batch]
            yield img_batch, mask_batch