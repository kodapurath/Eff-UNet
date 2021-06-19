import keras
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
BATCH_SIZE = 3
CLASSES = get_cityscapes_labels()
LR = 0.0001
EPOCHS = 10
BACKBONE = 'efficientnetb1'

preprocess_input = sm.get_preprocessing(BACKBONE)

# define network parameters
# n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)  # case for binary and multiclass segmentation
n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) )  # case for binary and multiclass segmentation
activation = 'sigmoid' if n_classes == 1 else 'softmax'

# create model
model = sm.Unet(BACKBONE, classes=n_classes, activation=activation)
print('created model ',model)
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
model.compile(optim, total_loss, metrics)
print('MODEL===========',model)
# print(model.summary())

# compile keras model with defined optimozer, loss and metrics

"""# Segmentation model training"""
import segmentation_models as sm
# define network parameters
# n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)  # case for binary and multiclass segmentation
print("Loading training dataset.....")
# Dataset for train images
train_dataset = Dataset(
    x_train_dir,
    y_train_dir,
    classes=CLASSES,
    augmentation=get_training_augmentation(),
    preprocessing=get_preprocessing(preprocess_input),
)
print("Loading validation dataset.....")
# Dataset for validation images
valid_dataset = Dataset(
    x_valid_dir,
    y_valid_dir,
    classes=CLASSES,
    augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(preprocess_input),
)
print("Training Dataloading ....")
train_dataloader = Dataloder(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
print("Validation Dataloading ....")
valid_dataloader = Dataloder(valid_dataset, batch_size=1, shuffle=False)

# check shapes for errors
print("DATALODAEER SHAPE 0==========================",train_dataloader[0][0].shape)
print("DATALODAEER SHAPE 1==========================",train_dataloader[0][1].shape)
print ("Actual 1======================",(BATCH_SIZE, 512, 512, n_classes))
assert train_dataloader[0][0].shape == (BATCH_SIZE, 512, 512, 3)
assert train_dataloader[0][1].shape == (BATCH_SIZE, 512, 512, n_classes)

# define callbacks for learning rate scheduling and best checkpoints saving
# /content/drive/MyDrive/Ottonomy/best_model.h5
callbacks = [
    keras.callbacks.ModelCheckpoint('../drive/MyDrive/Ottonomy/best_model.h5', save_weights_only=True, save_best_only=True, mode='min'),
    keras.callbacks.ReduceLROnPlateau(),
]

# train model
history = model.fit_generator(
    train_dataloader,
    steps_per_epoch=len(train_dataloader),
    epochs=EPOCHS,
    callbacks=callbacks,
    validation_data=valid_dataloader,
    validation_steps=len(valid_dataloader),
)

# Plot training & validation iou_score values
plt.figure(figsize=(30, 5))
plt.subplot(121)
plt.plot(history.history['iou_score'])
plt.plot(history.history['val_iou_score'])
plt.title('Model iou_score')
plt.ylabel('iou_score')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig("train_val_iou_score.jpg")
# Plot training & validation loss values
plt.subplot(122)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig("train_val_loss.jpg")
plt.show()
