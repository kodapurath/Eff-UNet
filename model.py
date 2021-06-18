import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt
from cityscapes_downloader import data_path_loader
from dataset import *
from augmentation import *
from dataloader import *
from labels import *
# Downloading cityscapes data
x_train_dir, y_train_dir, x_valid_dir, y_valid_dir, x_test_dir, y_test_dir = data_path_loader()
"""# Segmentation model training"""
import segmentation_models as sm


def Model():
    # sm.set_framework('tf.keras')
    # segmentation_models could also use `tf.keras` if you do not have Keras installed
    # or you could switch to other framework using `sm.set_framework('tf.keras')`

    BACKBONE = 'efficientnetb3'
    CLASSES = get_cityscapes_labels()
    LR = 0.0001

    preprocess_input = sm.get_preprocessing(BACKBONE)

    # define network parameters
    n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)  # case for binary and multiclass segmentation
    activation = 'sigmoid' if n_classes == 1 else 'softmax'

    # create model
    model = sm.Unet(BACKBONE, classes=n_classes, activation=activation)
    print('creatined model ',model)
    print(model.summary())
    # define optomizer
    optim = keras.optimizers.Adam(LR)

    # Segmentation models losses can be combined together by '+' and scaled by integer or float factor
    # set class weights for dice_loss (car: 1.; pedestrian: 2.; background: 0.5;)
    dice_loss = sm.losses.DiceLoss(class_weights=np.ones(len(CLASSES)))
    focal_loss = sm.losses.BinaryFocalLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss()
    total_loss = dice_loss + (1 * focal_loss)

    # actulally total_loss can be imported directly from library, above example just show you how to manipulate with losses
    # total_loss = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss

    metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
    print('MODEL before compiling ===========',model)
    model = model.compile(optim, total_loss, metrics)
    print('MODEL===========',model)
    # print(model.summary())

    # compile keras model with defined optimozer, loss and metrics
    return model,optim,total_loss,metrics,preprocess_input, metrics
