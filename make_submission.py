"""
Generate a submission file

Save your model with model.save('my_model.h5')
Alternatively, use the ModelCheckpoint callback 
(https://keras.io/callbacks/#modelcheckpoint)

Run this file as: 
$ python make_submission.py my_model.h5 test_dir

where test_dir is the folder with the test images
"""
import glob
import ntpath
import sys

import numpy as np
from scipy.misc import imresize, imread
from keras.models import load_model

def make_submission(model_file, test_dir):
    model = load_model(model_file)
    img_shape = model.layers[0].get_config()['batch_input_shape']
    image_files = glob.glob(test_dir + '/*.jpg')
    with open('submission.csv', 'w') as sub_file:
        sub_file.write('image,ALB,BET,DOL,LAG,NoF,OTHER,SHARK,YFT\n')
        for image_file in image_files:
            img = imread(image_file)
            # Resize to input shape of network
            img = imresize(img, img_shape[2:])
            # Put image in theano order
            img = np.transpose(img, [2, 0, 1]).reshape(1, *img_shape[1:])
            probs = model.predict_proba(img, batch_size=1, verbose=0)
            # Write predictions to file
            csv = '%s,%s\n' % (ntpath.basename(image_file), ','.join([str(f) for f in probs[0]]))
            sub_file.write(csv)



if __name__ == '__main__':
    if len(sys.argv) != 3:
        print 'Usage: python %s model test_dir' % __file__
    else:
        make_submission(sys.argv[1], sys.argv[2])