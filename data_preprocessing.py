import pandas as pd
import numpy as np

def data_process():
    train_data = pd.read_csv('train.csv')
    train_data = np.array(train_data)
    
    sigma = train_data.std(axis = 0, dtype='float')
    mean = np.mean(train_data, axis=0)

    test_data = pd.read_csv('test.csv')
    test_data = np.array(test_data)

    for row in train_data:
        for i in range(len(sigma)):
            if sigma[i] == 0:
                continue
            row[i] = ( row[i] - mean[i] ) / sigma[i]

    for row in test_data:
        for i in range(len(sigma)):
            if sigma[i] == 0:
                continue
            row[i] = ( row[i] - mean[i] ) / sigma[i]    

    train_data = pd.DataFrame(train_data)
    print(train_data)
    
data_process()
