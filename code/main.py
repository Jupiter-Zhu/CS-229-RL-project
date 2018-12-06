import numpy as np
from utils import load_dataset


inputs, labels=load_dataset(csv_path='./data/stock_data_apple.csv', label_col='Labels')
print(len(inputs) )
print(len(labels[labels!=10]))

