import keras
import numpy as np
import matplotlib.pyplot as plt
from cityscapes_downloader import data_path_loader
from dataset import *
from augmentation import *
from dataloader import *
from labels import *
from tensorflow.keras.models import load_model
import segmentation_models as sm
import efficientnet.tfkeras
from tensorflow.keras.models import load_model# import efficientnet.keras as k
# model=keras.models.load_model('../drive/MyDrive/Ottonomy/saved_model/my_model2')


current_epoch=8



BATCH_SIZE = 3
CLASSES = get_cityscapes_labels()
EPOCHS = 1
BACKBONE = 'efficientnetb1'
preprocess_input = sm.get_preprocessing(BACKBONE)

n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) ) 
dice_loss = sm.losses.DiceLoss(class_weights=np.ones(len(CLASSES)))
focal_loss = sm.losses.BinaryFocalLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)
metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

# model=load_model('../drive/MyDrive/Ottonomy/best_model-3.h5',custom_objects={'f1-score':sm.metrics.FScore(threshold=0.5),'dice_loss_plus_1focal_loss': total_loss,'iou_score':sm.metrics.IOUScore(threshold=0.5)})

# print(model.summary())

x_train_dir, y_train_dir, x_valid_dir, y_valid_dir, x_test_dir, y_test_dir = data_path_loader()

first_index=0
div=int(len(x_train_dir)/6)
last_index=div-1


while last_index<=len(x_train_dir)-1:

  x_train_dir=x_train_dir[first_index:last_index]
  y_train_dir=y_train_dir[first_index:last_index]
  first_index=last_index+1
  last_index+=div
  model=keras.models.load_model('../drive/MyDrive/Ottonomy/saved_model/my_model_ep-'+str(current_epoch-1),custom_objects={'f1-score':sm.metrics.FScore(threshold=0.5),'dice_loss_plus_1focal_loss': total_loss,'iou_score':sm.metrics.IOUScore(threshold=0.5)})
  train_dataset = Dataset(
      x_train_dir,
      y_train_dir,
      classes=CLASSES,
      augmentation=get_training_augmentation(),
      preprocessing=get_preprocessing(preprocess_input),
  )
  # print("Loading validation dataset.....")
  # Dataset for validation images


  # valid_dataset = Dataset(
  #     x_valid_dir,
  #     y_valid_dir,
  #     classes=CLASSES,
  #     augmentation=get_validation_augmentation(),
  #     preprocessing=get_preprocessing(preprocess_input),
  # )



  # print("Training Dataloading ....")
  train_dataloader = Dataloder(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
  # print("Validation Dataloading ....")



  # valid_dataloader = Dataloder(valid_dataset, batch_size=1, shuffle=False)

  # check shapes for errors
  # print("DATALODAEER SHAPE 0==========================",train_dataloader[0][0].shape)
  # print("DATALODAEER SHAPE 1==========================",train_dataloader[0][1].shape)
  # print ("Actual 1======================",(BATCH_SIZE, 512, 512, n_classes))
  assert train_dataloader[0][0].shape == (BATCH_SIZE, 512, 512, 3)
  assert train_dataloader[0][1].shape == (BATCH_SIZE, 512, 512, n_classes)

  # define callbacks for learning rate scheduling and best checkpoints saving
  # /content/drive/MyDrive/Ottonomy/best_model.h5
  callbacks = [
      keras.callbacks.ModelCheckpoint('../drive/MyDrive/Ottonomy/best_model_ep-'+str(current_epoch)+'.h5', save_weights_only=False, save_best_only=True, mode='min'),
      keras.callbacks.ReduceLROnPlateau(),
  ]

  # train model
  history = model.fit_generator(
      train_dataloader,
      steps_per_epoch=len(train_dataloader),
      epochs=EPOCHS,
      callbacks=callbacks
      # validation_data=valid_dataloader,
      # validation_steps=len(valid_dataloader),
  )
  model.save('../drive/MyDrive/Ottonomy/saved_model/my_model_ep-'+str(current_epoch))
  current_epoch+=1
