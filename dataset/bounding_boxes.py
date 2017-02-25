import json
from collections import defaultdict
import os
import glob

def get_bounding_boxes(bbox_folder):
    """
    Read bounding boxes from json files created by Sloth

    # Params
    - bbox_folder : folder containing json files with bounding boxes

    # Returns
    - A dictionary mapping filename to a list of bounding boxes
      of the form (x, y, width, height)
    """
    bboxes = defaultdict(list)
    file_paths = glob.glob(os.path.join(bbox_folder, '*.json'))
    for file_path in file_paths:
        with open(file_path) as file:
            data = json.load(file)
            for image in data:
                img_name = os.path.basename(image['filename'])
                for annot in image['annotations']:
                    # make sure that coordinates are valid
                    bbox = (max(0, int(round(annot['x']))),
                            max(0, int(round(annot['y']))),
                            int(round(annot['width'])),
                            int(round(annot['height'])),
                            annot['class'])
                    bboxes[img_name].append(bbox)
    return bboxes