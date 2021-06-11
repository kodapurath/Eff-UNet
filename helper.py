os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import os
import cv2
import keras
import numpy as np
import matplotlib.pyplot as plt
from cityscapes_downloader import data_path_loader
from labels import get_cityscapes_labels
from visualize import visualize, denormalize
from dataset import *
from augmentation import *
from dataloader import *
# Downloading cityscapes data
x_train_dir, y_train_dir, x_valid_dir, y_valid_dir, x_test_dir, y_test_dir = data_path_loader()
# Lets look at data we have

dataset = Dataset(x_train_dir, y_train_dir, classes=['road', 'sidewalk',
                                                     'sky', 'person', 'bicycle'])
print("DATASET = ", dataset)
image, mask = dataset[3]  # get some sample
print("IMAGE = ", image)
print("MASK = ", mask)
visualize(image=image, road_mask=mask[..., 0], sidewalk_mask=mask[..., 1].squeeze(),
          sky_mask=mask[..., 2].squeeze(), person_mask=mask[..., 3].squeeze(), bicycle_mask=mask[..., 4].squeeze())


# Lets look at augmented data we have
dataset = Dataset(x_train_dir, y_train_dir, classes=['car', 'sky'], augmentation=get_training_augmentation())

image, mask = dataset[12]  # get some sample
visualize(
    image=image,
    cars_mask=mask[..., 0].squeeze(),
    sky_mask=mask[..., 1].squeeze(),
    background_mask=mask[..., 2].squeeze(),
)


