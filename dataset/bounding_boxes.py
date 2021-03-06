import json
from collections import defaultdict
import os
import glob

import skimage.morphology as morph
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2
from scipy.ndimage.measurements import center_of_mass

from preprocess import load_image

def get_bounding_boxes(bbox_folder, resize=None, data_folder='', include_class=False):
    """
    Read bounding boxes from json files created by Sloth

    # Params
    - bbox_folder : folder containing json files with bounding boxes
    - resize : None or tuple (height, width). If set, the bounding box
               will be rescaled to this size.
    - data_folder: If resize is not None, this folder will be used for
                   getting the size of each image. This should be the
                   original training data.
    - include_class : If true, the fifth number returned will be the class

    # Returns
    - A dictionary mapping filename to a list of bounding boxes
      of the form (x, y, width, height)
    """
    bboxes = defaultdict(list)
    file_paths = glob.glob(os.path.join(bbox_folder, '*.json'))
    if len(file_paths) == 0:
        raise ValueError('No boundingboxes found in %s' % bbox_folder)
    for file_path in file_paths:
        with open(file_path) as file:
            data = json.load(file)
            for image in data:
                img_name = os.path.basename(image['filename'])
                for annot in image['annotations']:
                    x, y, width, height = annot['x'], annot['y'], annot['width'], annot['height']
                    label = annot['class']
                    if resize is not None:
                        img = load_image(os.path.join(data_folder, label, img_name))
                        size = np.array(img.shape[:2])
                        aspect = size.astype(np.float32) / np.array(resize).astype(np.float32)
                        x, width = x/aspect[1], width/aspect[1]
                        y, height = y/aspect[1], height/aspect[1]

                    # make sure that coordinates are valid
                    bbox = (max(0, int(round(annot['x']))),
                            max(0, int(round(annot['y']))),
                            int(round(annot['width'])),
                            int(round(annot['height'])),
                            annot['class'])
                    if not include_class:
                        bbox = bbox[:4]
                    bboxes[img_name].append(bbox)
    return bboxes

def bbox_from_segmentation(segm, threshold=0.9, padding=0, around_center=False):
    """
    Find a bounding box around the largest connected
    component in a thresholded segmentation

    # Params
    - segm : segmentation of shape (height, width)
    - threshold : threshold to apply to the segmentation
    - padding : pad the box by this amount of pixels on each side
    - around_center : If true, the center of mass of the segmentation
                      will be used as the center for the bounding box
                      with sides padding x padding

    # Returns
    - x, y, width, height bounding box coordinates
    """
    segm = segm > threshold
    labels, num_labels = morph.label(segm, return_num=True)
    if num_labels == 0:
        # no bounding box found
        return
   
    # Zero is background, so we ignore it
    # But we do have to +1 to negate this after argmax
    largest = np.argmax([np.sum(labels == label)
                            for label in range(1, num_labels+1)])
    segm = labels == largest + 1
    
    if around_center:
        y_center, x_center = center_of_mass(segm)
        side = padding*2
        x = int(round(x_center-padding))
        y = int(round(y_center-padding))
        x = max(0, x)
        y = max(0, y)
        return x, y, side, side

    # we find the top-left coordinate by finding the first
    # column and row with a 1
    x1 = np.argmax(np.max(segm, axis=0))
    y1 = np.argmax(np.max(segm, axis=1))
    # The bottom-right coordinate is found by 
    x2 = segm.shape[1] - np.argmax(np.max(np.fliplr(segm), axis=0))
    y2 = segm.shape[0] - np.argmax(np.max(np.flipud(segm), axis=1))

    # Pad the box if possible
    x1, y1 = max(0, x1-padding), max(0, y1-padding)
    x2, y2 = min(segm.shape[1], x2+padding), min(segm.shape[0], y2+padding)

    return x1, y1, x2-x1, y2-y1

def visualize_bbox(img, bounding_box, segm=None, img_size=(1280, 720)):
    """
    Visualize a bounding box in an image with optional segmentation

    # Params:
    - img: string of the filepath or numpy array
    - bounding_box: (x, y, width, height) or None
    - segm: numpy array of segmentation, same size as image
    - img_size: if a path is given for the image, the image will
                be resize to this (width, height)
    """
    if isinstance(img, basestring):
        img = load_image(img)
        img = cv2.resize(img, img_size, 
                         interpolation=cv2.INTER_LINEAR)
    
    plt.figure(figsize=(10, 7))
    plt.imshow(img)
    if segm is not None:
        plt.imshow(segm, alpha=0.3)
    
    if bounding_box is not None:
        (x, y, width, height) = bounding_box
        ax = plt.gca()
        ax.add_patch(
            Rectangle(
                (x, y),
                width,
                height,
                fill=False,
                edgecolor='red',
                linewidth=3
            )
        )
    else:
        print 'No bounding box'
    plt.show()
    
def get_largest_bbox(bboxes):
    """
    Get the largest bbox from a list of bboxes
    """
    sq_size = [bbox[2]*bbox[3] for bbox in bboxes]
    return bboxes[np.argmax(sq_size)]

def largest_bbox_per_image(bboxes):
    """
    Only keep the largest bbox per image

    # Params
    - bboxes : dictionary of img_name -> list of bboxes

    # Returns:
    Same as input, but with only 1 bbox
    """
    return {k:get_largest_bbox(v) for k, v in bboxes.iteritems()}
