import numpy as np
from sklearn.model_selection import train_test_split
import os
import glob
import collections
import pandas as pd
import sys

np.random.seed(4321)

fish_classes = ['ALB', 'BET', 'DOL', 'LAG', 'OTHER', 'SHARK', 'YFT']

if os.path.isfile('../../Data/validation_FishNoFish/train.csv') or os.path.isfile('../../Data/validation_FishNoFish/test.csv'):
    print 'Train and test files already exists'
    sys.exit(0)

files = []

for label in fish_classes:
    fish = []
    for filename in glob.glob('../../Data/train/{}/*'.format(label)):
        fish.append([filename[-13::], 'FISH'])
    np.random.shuffle(fish)
    fish = fish[0:70] # Only take 70 per fish class for a balanced dataset
    for im in fish:
        files.append(im) 
     
for filename in glob.glob('../../Data/train/NoF/*'):
    files.append([filename[-13::], 'NoF'])

files = np.array(files)

X_train, X_test, y_train, y_test = train_test_split(files[:, 0], files[:, 1],
                                                    test_size=.2)
train = pd.DataFrame.from_dict([[X_train[i], y_train[i]] for i in
                               range(len(X_train))])
test = pd.DataFrame.from_dict([[X_test[i], y_test[i]] for i in
                               range(len(X_test))])

train.to_csv('../data/validation_FishNoFish/train.csv',
             header=['filename', 'label'],
             index=False)
test.to_csv('../data/validation_FishNoFish/test.csv',
            header=['filename', 'label'],
            index=False)
