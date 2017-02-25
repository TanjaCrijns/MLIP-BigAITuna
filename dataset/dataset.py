import pandas as pd

labels = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
train = pd.read_csv('../data/validation/train.csv')
validation = pd.read_csv('../data/validation/test.csv')