import numpy as np
import matplotlib.pyplot as plt
from cityscapes_downloader import data_path_loader
from labels import get_cityscapes_labels
from visualize import visualize, denormalize
from dataset import *
from augmentation import *
from dataloader import *
from model import Model
import cv2
from colorizer_t import *
# Downloading cityscapes data

x_train_dir, y_train_dir, x_valid_dir, y_valid_dir, x_test_dir, y_test_dir = data_path_loader()
x_test_dir=x_test_dir[:1]
y_test_dir=y_test_dir[:1]
"""Model testing"""
import segmentation_models as sm
BATCH_SIZE = 3
CLASSES = get_cityscapes_labels()
LR = 0.0001
EPOCHS = 1
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


test_dataset = Dataset(
    x_test_dir,
    y_test_dir,
    classes=get_cityscapes_labels(),
    # augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(preprocess_input),
)

test_dataloader = Dataloder(test_dataset, batch_size=1, shuffle=False)

# load best weights
model.load_weights('best_model.h5')

scores = model.evaluate_generator(test_dataloader)

print("Loss: {:.5}".format(scores[0]))
for metric, value in zip(metrics, scores[1:]):
    print("mean {}: {:.5}".format(metric.__name__, value))

"""# Visualization of results on test dataset"""

n = 1
ids = np.random.choice(np.arange(len(test_dataset)), size=n)
print("IDS======",ids)
for i in ids:
    image, gt_mask = test_dataset[i]
    print("Test image shape ======== ",image.shape)
    cv2.imwrite("qwqw.jpg",image)
    print(image.shape)
    print(gt_mask.shape)
    image = np.expand_dims(image, axis=0)
    pr_mask = model.predict(image)
    # pr_mask = np.expand_dims(pr_mask, axis=0)
    print(pr_mask.shape)
    pr_mask=np.squeeze(pr_mask)
    print(pr_mask.shape)
    print(np.max(np.unique(pr_mask)))
    print(np.min(np.unique(pr_mask)))
    im=pr_mask
    h=pr_mask.shape[0]
    w=pr_mask.shape[1]
    colorizer_head(im,n_classes,h, w)
    # stack = np.concatenate([image, pr_mask], axis=-1)
    # cv2.imwrite('infered.png',stack)
    # cv2.imshow("predictions", stack)
    # cv2.waitKey(1)
    # visualize(i,
    #     image=denormalize(image.squeeze()),
    #     gt_mask=gt_mask.squeeze(),
    #     pr_mask=pr_mask.squeeze()
    # )
