import os
import glob
import cv2
import numpy as np
import pandas as pd
from sklearn import preprocessing


def get_dataset(data_folder_path, color_scheme):
    '''
        Make training dataset, testing dataset and validation dataset
    '''
    X_train = convert_images_to_numpy_array(
        data_folder_path, "train/*", color_scheme)
    X_test = convert_images_to_numpy_array(
        data_folder_path, "test/*", color_scheme)
    X_val = convert_images_to_numpy_array(
        data_folder_path, "val/*", color_scheme)

    y_test = pd.read_csv(os.path.join(
        data_folder_path, "labels_test.csv"), header=None)
    y_train = pd.read_csv(os.path.join(
        data_folder_path, "labels_train.csv"), header=None)
    y_val = pd.read_csv(os.path.join(
        data_folder_path, "labels_val.csv"), header=None)

    le = preprocessing.LabelEncoder()
    le.fit(y_train)
    y_train = le.transform(y_train)
    y_test = le.transform(y_test)
    y_val = le.transform(y_val)

    return X_train, X_test, X_val, y_train, y_test, y_val


def convert_images_to_numpy_array(data_folder_path, folder_name, color_scheme):
    '''
        Convert images for a folder to numpy arrays
    '''
    X = []
    files = glob.glob(os.path.join(data_folder_path, folder_name))
    for _file in files:
        image = cv2.imread(_file)
        if color_scheme == "grayscale":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        X.append(image)
    return np.asarray(X)
