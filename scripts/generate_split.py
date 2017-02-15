import numpy as np
from sklearn.model_selection import train_test_split
import os
import glob
import collections
import pandas as pd
import sys

np.random.seed(1234)

classes = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']

if os.path.isfile('../data/validation/train.csv') or os.path.isfile('../data/validation/test.csv'):
    print 'Train and test files already exists'
    sys.exit(0)

files = []

for label in classes:
    for filename in glob.glob('../data/train/{}/*'.format(label)):
        files.append([filename[-13::], label])

files = np.array(files)

X_train, X_test, y_train, y_test = train_test_split(files[:, 0], files[:, 1],
                                                    test_size=.2)
train = pd.DataFrame.from_dict([[X_train[i], y_train[i]] for i in
                               range(len(X_train))])
test = pd.DataFrame.from_dict([[X_test[i], y_test[i]] for i in
                               range(len(X_test))])

train.to_csv('../data/validation/train.csv',
             header=['filename', 'label'],
             index=False)
test.to_csv('../data/validation/test.csv',
            header=['filename', 'label'],
            index=False)
