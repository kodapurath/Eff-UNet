
import numpy as np
import matplotlib.pyplot as plt
from cityscapes_downloader import data_path_loader
from labels import get_cityscapes_labels
from visualize import visualize, denormalize
from dataset import *
from augmentation import *
from dataloader import *
from model import Model

"""# Model Evaluation"""
x_train_dir, y_train_dir, x_valid_dir, y_valid_dir, x_test_dir, y_test_dir = data_path_loader()
import segmentation_models as sm

# BACKBONE = 'efficientnetb3'
model, preprocess_input, metrics = Model()

test_dataset = Dataset(
    x_test_dir,
    y_test_dir,
    classes=get_cityscapes_labels(),
    augmentation=get_validation_augmentation(),
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

n = 5
ids = np.random.choice(np.arange(len(test_dataset)), size=n)

for i in ids:
    image, gt_mask = test_dataset[i]
    image = np.expand_dims(image, axis=0)
    pr_mask = model.predict(image)

    visualize(
        image=denormalize(image.squeeze()),
        gt_mask=gt_mask.squeeze(),
        pr_mask=pr_mask.squeeze(),
    )
