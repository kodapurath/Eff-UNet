import cv2
import numpy as np
from labels import get_color_map

def colorizer_head(n_classes,h, w):
    print("No of Classes = ",n_classes)
    print("Height = ",h)
    print("Width = ",w)
    im = np.random.rand(n_classes,h,w)
    # print(im)
    # print(im.shape)
    # print("Along axis 0")
    # print(im.max(axis=0))
    print("Indices of max axis 0")
    max_channel=im.argmax(axis=0)
    print(max_channel)
    print("Shape of Indices of max axis 0")
    print(max_channel.shape)
    im_new = np.zeros((3,h,w))
    # print(im_new)
    # print(im_new.shape)
    color_map=get_color_map()
    # im_new=[im_new[:,i,j]= for ]
    for i in range(h):
        for j in range(w):
            # print(i,j)
            im_new[:,i,j]=color_map[max_channel[i,j]][1]
            # print(max_channel[i,j])
            # print(color_map[max_channel[i,j]][1])
    print(im_new)#RGB Mask
    print("Mask image shape = ",im_new.shape)
    im_new=im_new.reshape(h, w, 3)
    cv2.imwrite('color_img.jpg', im_new)
    cv2.imshow("image", im_new)
    cv2.waitKey()
colorizer_head(33,1000,700)#n_classes,height,width
