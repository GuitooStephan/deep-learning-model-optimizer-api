import os
import uuid
import cv2
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split


def save_images(data_folder_path, folder_name, data):
    '''
        Save each row as image in a folder in base project path
    '''
    if not os.path.exists(os.path.join(data_folder_path, folder_name)):
        os.makedirs(os.path.join(data_folder_path, folder_name))
    for i in range(data.shape[0]):
        reshape_image = data[i].reshape(28, 28, 1)
        cv2.imwrite(os.path.join(data_folder_path, folder_name,
                    str(i) + ".jpg"), reshape_image)


def save_labels_in_csv(data_folder_path, file_name, data):
    '''
        Save labels in a csv file
    '''
    df = pd.DataFrame(data, columns=['label'])
    df.to_csv(
        os.path.join(data_folder_path, file_name),
        index=False, header=False
    )


cwd = Path(os.getcwd()).parent.parent.absolute()
fashion_mnist_folder_path = os.path.join(cwd, "fashion-mnist-data")
# Create a project folder in Documents/optimizer
base_project_path = os.path.join(
    os.path.expanduser("~"), "Documents", "optimizer"
)

# Read train and validation data
train_data = pd.read_csv(os.path.join(
    fashion_mnist_folder_path, "fashion-mnist_train.csv"))
val_data = pd.read_csv(os.path.join(
    fashion_mnist_folder_path, "fashion-mnist_test.csv"))

# Split X and y
X_train = train_data.iloc[:, 1:].values
y_train = train_data.iloc[:, 0].values
X_val = val_data.iloc[:, 1:].values
y_val = val_data.iloc[:, 0].values

# Split the train data to train, test
X_train, X_test, y_train, y_test = train_test_split(
    X_train, y_train, test_size=0.2, random_state=0)

project_id = str(uuid.uuid4())
save_images(
    os.path.join(base_project_path, project_id, 'data'),
    "train",
    X_train
)
save_labels_in_csv(
    os.path.join(base_project_path, project_id, 'data'),
    "labels_train.csv",
    y_train
)

save_images(
    os.path.join(base_project_path, project_id, 'data'),
    "test",
    X_test
)
save_labels_in_csv(
    os.path.join(base_project_path, project_id, 'data'),
    "labels_test.csv",
    y_test
)

save_images(
    os.path.join(base_project_path, project_id, 'data'),
    "val",
    X_val
)
save_labels_in_csv(
    os.path.join(base_project_path, project_id, 'data'),
    "labels_val.csv",
    y_val
)
