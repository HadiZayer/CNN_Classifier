import torch
import numpy as np
from torch.autograd import Variable
import os
import imageio

def load_training_data(data_dir, num_classes, validation_set_percentage=0.2):
    train_data, train_labels, validation_data, validation_labels = [], [], [], []
    for i in range(num_classes):
        current_path = os.path.join(data_dir, str(i))
        for root, dirs, files in os.walk(current_path):
            for file in files:
                filepath = os.path.join(root, file)
                if(file.endswith('.png')):
                    img = imageio.imread(filepath)
                    if(np.random.rand() > validation_set_percentage):
                        train_data.append(img)
                        train_labels.append(i)
                    else:
                        validation_data.append(img)
                        validation_labels.append(i)
                        
    train_data = np.array(train_data)
    train_labels = np.array(train_labels)
    
    train_data = torch.from_numpy(train_data)
    train_labels = torch.from_numpy(train_labels)
    
    validation_data = np.array(validation_data)
    validation_labels = np.array(validation_labels)
    
    validation_data = torch.from_numpy(validation_data)
    validation_labels = torch.from_numpy(validation_labels)
    
    return train_data, train_labels, validation_data, validation_labels

def load_test_data(data_dir):
    test_data, test_id = [], []   
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            filepath = os.path.join(root, file)
            if(file.endswith('.png')):
                img = imageio.imread(filepath)
                test_data.append(img)
                test_id.append(file.replace('.png', ''))

                        
    test_data = np.array(test_data)
    
    test_data = torch.from_numpy(test_data)
    
    return test_data, test_id