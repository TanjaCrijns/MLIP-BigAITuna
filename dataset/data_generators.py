from __future__ import division

import os

import numpy as np
from keras.utils.np_utils import to_categorical as onehot

from preprocess import *
from dataset import labels
from bounding_boxes import bbox_from_segmentation

def get_data(data_df, data_folder, labels, batch_size=32, shuffle=True, bboxes=None,
             augmentation=True, img_size=(256, 256), balance_batches=False, **kwargs):
    """
    Generator to train a model on images.

    # Params
    - data_df : DataFrame of filename and label for each image
    - data_folder : folder where data resides. Should have structure
                    `data_folder/label/img_name`
    - labels : list of labels
    - batch_size : number of images per batch
    - shuffle : present images in random order (each epoch)
    - bboxes : A dictionary img_name -> (x, y, width, height). If this
               is given, the image will be cropped to this region.
    - augmentation : perform data augmentation
    - img_size : sample patches of this size from the image and mask
    - balance_batches : If true, balances batches so each class is 
                        equally represented
    - kwargs : passed to the preprocess function

    # Returns
    - batch of images (batch_size, 3, img_size[0], img_size[1])
    - batch of labels (batch_size, len(labels))
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
        labelcount = np.zeros(len(labels))
        while i < n:
            img_batch = np.zeros((batch_size, 3) + img_size, dtype=np.float32)
            label_batch = np.zeros((batch_size, n_classes), dtype=np.uint8)
            j = 0
            label_count = np.zeros(len(labels))
            while j < batch_size:
                img_name, label = data[i]
                i += 1
                lab_nr = labels.index(label)
                
                if balance_batches and label_count[lab_nr] >= batch_size / len(labels):
                    continue
                label_count[lab_nr] += 1
                img_path = os.path.join(data_folder, label, img_name)
                img = load_image(img_path)
                if bboxes is not None:
                    if img_name in bboxes:
                        boxes = bboxes[img_name]
                        if isinstance(boxes, list):
                            # if we have a list of boxes, choose a box at random
                            box_idx = np.random.randint(len(boxes))
                            x, y, width, height = boxes[box_idx]
                        else:
                            # otherwise it already is just 1 box
                            x, y, width, height = boxes
                    else :
                        # no bounding box found, choose random box
                        x = np.random.randint(img.shape[1] - 256)
                        y = np.random.randint(img.shape[0] - 256)
                        height, width = img_size

                    # Crop the image to the bounding box
                    x, y, width, height = [int(n) for n in [x, y, width, height]]
                    img = img[y:y+height, x:x+width]
                img = preprocess(img, target_size=img_size, 
                                augmentation=augmentation, 
                                zero_center=True, scale=1./255.,
                                **kwargs)
                img_batch[j] = img
                label_batch[j] = onehot(labels.index(label), len(labels))
                j += 1
            yield img_batch, label_batch

def get_test_data(files, batch_size=4, img_size=(720, 1280), **kwargs):
    """
    Generator for test data

    # Params
    - files : list of files (e.g., output of glob)
    - batch_size : number of images per batch
    - img_size : size to resize images to (height, width)
    - kwargs : passed to the preprocess function

    # Returns
    - batches of (batch_size, 3, img_size[0], img_size[1])
    """
    i = 0
    n = len(files)
    # cycle to avoid batches not lining up with dataset size
    files = files + files
    while True:
        batch = np.zeros((batch_size, 3) + img_size)
        for j in range(batch_size):
            img = load_image(files[i])
            img = preprocess(img, target_size=img_size, augmentation=False, 
                             zero_center=True, scale=1./255., **kwargs)
            batch[j] = img
            i = (i + 1) % n
        yield batch
            

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
                img_name, label = data[i]
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
            yield img_batch, [label_batch, mask_batch]
            # yield img_batch, mask_batch

def get_data_with_bbox_coords(data_df, data_folder, bboxes, batch_size=32, shuffle=True, 
                              augmentation=True, img_size=(256, 256), **kwargs):
    """
    Generator to train a model on images.
    Images which don't have a bounding box are simply skipped.

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
    - batch of labels (batch_size, len(labels))
    """
    n_coords = 4
    
    while True:
        data = zip(data_df.filename.values, data_df.label.values)
        data = [(img, label) for img, label in data if img in bboxes and len(bboxes[img])!=0]
        n = len(data)
        if shuffle:
            data = np.random.permutation(data)
        data = list(data)

        # Double to allow for larger batch sizes
        data += data
        i = 0
        while i < n:
            img_batch = np.zeros((batch_size, 3) + img_size, dtype=np.float32)
            label_batch = np.zeros((batch_size, n_coords), dtype=np.int32)
            for j in range(batch_size):
                img_name, label = data[i]
                img_path = os.path.join(data_folder, label, img_name)
                img = load_image(img_path)
                
                bbox = bboxes[img_name]
                if len(bbox) == 5:
                    bbox = bbox[:4]
                x, y, width, height = bbox
                # Make a mask from the bounding box, so we can apply
                # data augmentation to it. Later we will convert it back
                mask = np.zeros(img.shape[:2], dtype=np.uint8)
                mask[y:y+height, x:x+width] = 1

                img, mask = preprocess(img, target_size=img_size,
                                       augmentation=augmentation,
                                       zero_center=True, scale=1./255.,
                                       mask=mask, **kwargs)
                
                img_batch[j] = img
                label_batch[j] = bbox_from_segmentation(mask)
                i += 1
            yield img_batch, label_batch
