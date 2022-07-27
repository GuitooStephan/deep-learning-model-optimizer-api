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
    X_train, y_train = create_x_y(
        data_folder_path, "train", "labels_train.csv", color_scheme)
    X_test, y_test = create_x_y(
        data_folder_path, "test", "labels_test.csv", color_scheme)
    X_val, y_val = create_x_y(
        data_folder_path, "val", "labels_val.csv", color_scheme)

    le = preprocessing.LabelEncoder()
    le.fit(y_train)
    y_train = le.transform(y_train)
    y_test = le.transform(y_test)
    y_val = le.transform(y_val)

    return X_train, X_test, X_val, y_train, y_test, y_val


def create_x_y(data_folder_path, images_folder_name, labels_file_name, color_scheme):
    '''
        Create numpy arrays of predictors and labels from images folder and labels file
    '''
    X = []

    images_folder_path = os.path.join(data_folder_path, images_folder_name)
    labels_file_path = os.path.join(data_folder_path, labels_file_name)
    labels_df = pd.read_csv(labels_file_path)

    for index in labels_df.index:
        image_name = labels_df.loc[index, 'id']
        _file = os.path.join(images_folder_path, f"{image_name}.jpg")
        image = cv2.imread(_file)
        if color_scheme == "grayscale":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        X.append(image)

    y = labels_df['label']
    return np.asarray(X), y
