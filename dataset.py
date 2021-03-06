import os
import cv2
import numpy as np
from labels import get_cityscapes_labels


# classes for data loading and preprocessing
class Dataset:
    CLASSES = get_cityscapes_labels()
    def __init__(self, images_dir, masks_dir, classes=None, augmentation=None, preprocessing=None):
        self.ids=[ im_id.split('/')[-1] for im_id in images_dir]
        self.images_fps = images_dir
        self.masks_fps = masks_dir
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
#         print('from dataset.py/Dataset class values or indices : ',self.class_values)
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
#         print("This is dataset.py/__get__item = ",i)
        image = cv2.imread(self.images_fps[i])
#         print("Original image size : ",image.shape)
        image = cv2.resize(image, (0,0), fx=0.5, fy=0.5)
#         print("Resized image size : ",image.shape)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)
        mask = cv2.resize(mask, (0,0), fx=0.5, fy=0.5)
#         print("MASK ADDRESS : ", self.masks_fps[i])
#         print("IMAGE ADDRESS : ", self.images_fps[i])
        # print("IMAGE INSIDE CLASS", image)
        # print("MASK INDSIDE CLASS", mask)
        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
#         print("inside dataset.py MASK SHAPE =============",mask.shape)
        # add background if mask is not binary
        # if mask.shape[-1] != 1:
        #     background = 1 - mask.sum(axis=-1, keepdims=True)
        #     mask = np.concatenate((mask, background), axis=-1)

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            # print("dataset.py/preprocessing")
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.ids)
