import keras
import numpy as np
import matplotlib.pyplot as plt
from cityscapes_downloader import data_path_loader
from dataset import *
from augmentation import *
from dataloader import *
from labels import *
from tensorflow.keras.models import load_model
from keras.callbacks import CSVLogger
# Downloading cityscapes data
x_train_dir, y_train_dir, x_valid_dir, y_valid_dir, x_test_dir, y_test_dir = data_path_loader()
# x_train_dir=x_train_dir[:100]
# y_train_dir=y_train_dir[:100]
# x_valid_dir=x_valid_dir[:100]
# y_valid_dir=y_valid_dir[:100]

"""# Segmentation model training"""
import segmentation_models as sm
BATCH_SIZE = 3
CLASSES = get_cityscapes_labels()
# LR = 0.001
EPOCHS = 5
BACKBONE = 'efficientnetb1'

preprocess_input = sm.get_preprocessing(BACKBONE)

# define network parameters
# n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)  # case for binary and multiclass segmentation
n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) )  # case for binary and multiclass segmentation
activation = 'sigmoid' if n_classes == 1 else 'softmax'

# create model
# model = sm.Unet(BACKBONE, classes=n_classes, activation=activation)
# optim = keras.optimizers.Adam(LR)
total_loss=sm.losses.CategoricalCELoss()
metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
# model.compile(optim, total_loss, metrics)
# model=load_model('../drive/MyDrive/Ottonomy/best_model1_256x256.h5')
# model=keras.models.load_model('../drive/MyDrive/Ottonomy/saved_model/my_model_2_July_256x256')
model=keras.models.load_model('../drive/MyDrive/Ottonomy/saved_model/my_model_2_July_256x256',
custom_objects={'f1-score':sm.metrics.FScore(threshold=0.5),'iou_score':sm.metrics.IOUScore(threshold=0.5)})

print('MODEL===========',model)
print(model.summary())

# compile keras model with defined optimozer, loss and metrics

"""# Segmentation model training"""
# import segmentation_models as sm
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
valid_dataloader = Dataloder(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

# check shapes for errors
print("DATALODAEER SHAPE 0==========================",train_dataloader[0][0].shape)
print("DATALODAEER SHAPE 1==========================",train_dataloader[0][1].shape)
print ("Actual 1======================",(BATCH_SIZE, 256, 256, n_classes))
assert train_dataloader[0][0].shape == (BATCH_SIZE, 256, 256, 3)
assert train_dataloader[0][1].shape == (BATCH_SIZE, 256, 256, n_classes)

class printlearningrate(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        optimizer = self.model.optimizer
        lr = keras.backend.eval(optimizer.lr)
        Epoch_count = epoch + 1
        print("Epoch:", Epoch_count)
        print("Learning Rate : ",lr)

printlr = printlearningrate() 

def scheduler(epoch):
  optimizer = model.optimizer
  return keras.backend.eval(optimizer.lr*0.9)

updatelr = keras.callbacks.LearningRateScheduler(scheduler)

# define callbacks for learning rate scheduling and best checkpoints saving
# /content/drive/MyDrive/Ottonomy/best_model.h5

csv_logger = CSVLogger("../drive/MyDrive/Ottonomy/model_history_log.csv", append=True)

callbacks = [
    keras.callbacks.ModelCheckpoint('../drive/MyDrive/Ottonomy/best_model_2_2_July_256x256.h5', save_weights_only=True, save_best_only=True, mode='min'),
    keras.callbacks.ReduceLROnPlateau(),printlr,updatelr,csv_logger
]


print("Learning Rate == ",keras.backend.eval(model.optimizer.lr))

# train model
history = model.fit_generator(
    train_dataloader,
    steps_per_epoch=len(train_dataloader),
    epochs=EPOCHS,
    callbacks=callbacks,
    validation_data=valid_dataloader,
    validation_steps=len(valid_dataloader),
)
model.save('../drive/MyDrive/Ottonomy/saved_model/my_model_2_July_256x256')

