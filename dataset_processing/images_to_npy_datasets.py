import os
import random

import numpy as np
from PIL import Image
from numpy import genfromtxt

SOURCE_FOLDER = "/home/pasca/School/Licenta/Datasets/all_movies_frames"
DESTINATION_FOLDER = "/home/pasca/School/Licenta/Datasets/all_movies_numpy"
LABELS_CSV_PATH = "/jet/prs/workspace/images/CelebA/original_labels/list_attr_celeba.csv"


def save_to_npy(files, destination_file, image_size):
    images = np.array([np.array(Image.open(fname)) for fname in files])
    images = images / 255
    # ratio_diff = np.abs(images.shape[1] - images.shape[2])
    # if images.shape[1] > images.shape[2]:
    #     images = images[:, ratio_diff // 2:-ratio_diff // 2, :]
    # else:
    #     images = images[:, :, ratio_diff // 2:-ratio_diff // 2]
    # images_resized = np.array([scipy.misc.imresize(image, [image_size, image_size]) for image in images])
    np.save(destination_file, images)


def split_dataset(source_folder, destination_folder, test, validation, image_size=64, shuffle=True, seed=None):
    files = [os.path.join(source_folder, f) for f in os.listdir(source_folder) if
             os.path.isfile(os.path.join(source_folder, f))]
    files.sort()
    indexes = [i for i in range(len(files))]

    if shuffle:
        if seed is not None:
            random.seed(seed)
        random.shuffle(indexes)

    files = [files[indexes[i]] for i in range(len(indexes))]
    nr_test = round(len(files) * test)
    nr_validation = round(len(files) * validation)

    test_files = files[:nr_test]
    validation_files = files[nr_test:nr_test + nr_validation]
    train_files = files[nr_test + nr_validation:]

    save_to_npy(test_files, os.path.join(destination_folder, "test.npy"), image_size)
    save_to_npy(validation_files, os.path.join(destination_folder, "validation.npy"), image_size)
    save_to_npy(train_files, os.path.join(destination_folder, "train.npy"), image_size)


def split_labels_dataset(csv_path, destination_folder, test, validation, shuffle=True, seed=None):
    labelsArray = genfromtxt(csv_path, delimiter=',')[1:, 1:]
    indexes = [i for i in range(len(labelsArray))]
    if shuffle:
        if seed is not None:
            random.seed(seed)
        random.shuffle(indexes)
    labelsArray = np.array([labelsArray[indexes[i]] for i in range(len(indexes))])
    test_labels = labelsArray[:test]
    validation_labels = labelsArray[test:test + validation]
    train_labels = labelsArray[test + validation:]
    np.save(os.path.join(destination_folder, "test_labels.npy"), test_labels)
    np.save(os.path.join(destination_folder, "validation_labels.npy"), validation_labels)
    np.save(os.path.join(destination_folder, "train_labels.npy"), train_labels)


def main():
    split_dataset(SOURCE_FOLDER, DESTINATION_FOLDER, test=0.15, validation=0.15, image_size=64, seed=42)
    # split_labels_dataset(LABELS_CSV_PATH, DESTINATION_FOLDER, test=100, validation=100, seed=42)


if __name__ == "__main__":
    main()
