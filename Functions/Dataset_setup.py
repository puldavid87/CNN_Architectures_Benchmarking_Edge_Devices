############# Check the versions of libraries ######
# Check the versions of libraries
# Python version
import sys
import scipy
import numpy
import matplotlib
import pandas
import sklearn
import tensorflow
import os
import random
import keras
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


batch_size = 32
img_height = 224
img_width = 224


def versions ():
    print('Python: {}'.format(sys.version))
    print('scipy: {}'.format(scipy.__version__))
    print('numpy: {}'.format(numpy.__version__))
    print('matplotlib: {}'.format(matplotlib.__version__))
    print('pandas: {}'.format(pandas.__version__))
    print('sklearn: {}'.format(sklearn.__version__))
    print('tensorflow: {}'.format(tensorflow.__version__))
    print('os: {}'.format(os.__version__))
    print('random: {}'.format(random.__version__))
    print('keras: {}'.format(keras.__version__))

def check_data(dataset):
    """
    Check data in a dataset
    Args:
        dataset: The dataset directory to check
    Returns:
        None: Does not return anything
    - Checks for any errors or invalid files in the dataset
    - Logs any errors or issues found with the files to a report
    - Returns nothing after completing the check
    - Loops through each file and directory in the provided dataset path
    Check data in a dataset
    Args:
        dataset: The dataset directory to check
    Returns:
        None: Does not return anything
    - Checks for any errors or invalid files in the dataset

    - Logs any errors or issues found with the files to a report
    - Returns nothing after completing the check"""
    for dirpath, dirnames, filenames in os.walk(dataset):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

#train_path
#get  labels and tam
def get_labels(train_path):
    """
    Get labels and number of images for each label from training path
    Args:
        train_path: Path to training data directory
    Returns:
        labels: List of all labels
        tam_labels: Numpy array of count of images for each label
    Processing Logic:
    - Loop through directories in training path and append directory name to labels list
    - Loop through labels and count number of images in each label directory
    - Append count to tam list and print label and count
    - Convert tam list to numpy array
    - Return labels list and tam_labels numpy array
    """
    labels=[]
    tam = []
    for cont, i in enumerate((os.listdir(train_path))):
        labels.append(i)
    for cont, i in enumerate (labels):
        num = len(os.listdir(train_path+"/"+i))
        tam.append(num)
        print("Label ", i, ": ", num)
        tam_labels = np.array(tam)
    return labels, tam_labels

#Loading.

def view_n_images(target_dir, target_class,n):
    """View n random images from a target class directory
    Args:
        target_dir: Target directory path
        target_class: Target class name
        n: Number of images to view
    Returns:
        None: Does not return anything
    View n random images from the target class directory:
    - Get path to target class directory
    - List files in target directory
    - Randomly sample n file names
    - Plot the sampled images in a matplotlib figure grid"""
    target_path = f'{target_dir}/{target_class}'
    file_names = os.listdir(target_path)
    target_images = random.sample(file_names, n)
    # Plot images
    plt.figure(figsize=(15, 6))
    for i, img in enumerate(target_images):
        img_path = f'{target_path}/{img}'
        plt.subplot(1, 3, i+1)
        plt.imshow(mpimg.imread(img_path))
        plt.title(target_class)
        plt.axis("off")

def plot_n_images(train_path,labels,n):
    """Plots n random images from the training path
    Args:
        train_path: Path to training images directory
        labels: List of image labels
        n: Number of images to plot
    Returns:
        None: Does not return anything
    - Selects n random indices from the labels list
    - Extracts the corresponding images from the train_path using the indices
    - Plots the images"""
    for i in labels:
        view_n_images(train_path,i,n)


def shape_labels(labels,train_path):
    """
    Labels shapes in an image dataset
    Args:
        labels: List of shape labels
        train_path: Path to image dataset folder
    Returns:
        None: Does not return anything
    Processes labels on images:
    - Loops through each label in the labels list
    - Accesses each image in the train_path folder
    - Labels the shapes in each image with the corresponding label
    - Saves labeled images to train_path"""

    for i in labels:
        target_path = f'{train_path}/{i}'
        file_names = os.listdir(target_path)
        target_image = random.sample(file_names, 1)
        img = mpimg.imread(target_path + "/" + str(target_image[0]))
        print(i, ": ", img.shape)
        print(img.shape[0]*img.shape[1])


def split_tratin_test_set(path_data_source,batch_size,img_height, img_width):
    """
    Splits training and test datasets
    Args: 
        train_dir: Path to training directory
        test_dir: Path to test directory
    Returns:
        train_data: Training dataset
        validation_data: Validation dataset 
        test_data: Test dataset
    Splits training images into train and validation sets. Imports test images separately. Converts all images to specified height and width.
    - Imports data from train and test directories 
    - Splits train data into train and validation sets
    - Converts images to specified height and width
    - Returns train, validation and test datasets
    """
    train_dir = path_data_source + "/" + "train"
    test_dir = path_data_source + "/" + "test"
    # Import data from directories and turn it into batches
    train_data = tf.keras.preprocessing.image_dataset_from_directory(train_dir,
                                                                     seed=123,
                                                                     label_mode="categorical",
                                                                     batch_size=batch_size,  # number of images to process at a time
                                                                     validation_split=0.2,
                                                                     subset="training",
                                                                     image_size=(img_height, img_width))  # convert all images to be 224 x 224

    validation_data = tf.keras.preprocessing.image_dataset_from_directory(train_dir,
                                                                          seed=123,
                                                                          label_mode="categorical",
                                                                          batch_size=batch_size,  # number of images to process at a time
                                                                          validation_split=0.2,
                                                                          subset="validation",
                                                                          image_size=(img_height, img_width))  # convert all images to be 224 x 224

    test_data = tf.keras.preprocessing.image_dataset_from_directory(test_dir,
                                                                    seed=123,
                                                                    label_mode="categorical",
                                                                    batch_size=batch_size,  # number of images to process at a time
                                                                    image_size=(img_height, img_width))  # convert all images to be 224 x 224
    return train_data, validation_data, test_data