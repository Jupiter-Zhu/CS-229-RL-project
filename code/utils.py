import numpy as np


def load_dataset(csv_path):
    """Load dataset from a CSV file.

    Args:
         csv_path: Path to CSV file containing dataset.
         label_col: Name of column to use as labels (should be 'y' or 'l').
         add_intercept: Add an intercept entry to x-values.

    Returns:
        xs: Numpy array of x-values (inputs).
        ys: Numpy array of y-values (labels).
    """

    
    # Load headers
    # with open(csv_path, 'r') as csv_fh:
    #     headers = csv_fh.readline().strip().split(',')
    # print(headers)

    # Load features and labels
    # x_cols = [i for i in range(len(headers)) if headers[i].startswith('Data')]
    # print(x_cols)
    # l_cols = [i for i in range(len(headers)) if headers[i] == label_col]
    # print(l_cols)
    inputs = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=(1,2,3,4,5))
     

    

    

    return inputs