"""
This is not the file you want to use. Go to cityscapes_data.py
"""




import os
def cityscapes_dataloader():
    DATA_DIR = 'data/cityscapes2/'
    cityscapes_username = 'abhay.k-oric@ottonomy.io'
    cityscapes_password = 'asjaanya'
    if not os.path.exists(DATA_DIR):
        print('Loading data...')
        os.system(
            'wget --keep-session-cookies --save-cookies=cookies.txt --post-data \'username=' + cityscapes_username + '&password=' + cityscapes_password + '&submit=Login\' https://www.cityscapes-dataset.com/login/')
        os.system(
            'wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=1')
        os.system(
            'wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=3')
        os.system('mkdir -p ' + DATA_DIR)
        os.system('unzip leftImg8bit_trainvaltest.zip -d ' + DATA_DIR)
        os.system('unzip gtFine_trainvaltest.zip -d ' + DATA_DIR)

    x = os.path.join(DATA_DIR, 'leftImg8bit_trainvaltest/leftImg8bit')
    y = os.path.join(DATA_DIR, 'gtFine_trainvaltest/gtFine')

    train_x = []
    path_leftimg = x
    for trainvaltest in os.listdir(path_leftimg):
        path_trainvaltest = os.path.join(path_leftimg, trainvaltest)
        for city in sorted(os.listdir(path_trainvaltest)):
            # print(city)
            path_image = os.path.join(path_trainvaltest, city)
            if trainvaltest == 'train':
                print(city)
                for image in sorted(os.listdir(path_image)):
                    train_x = train_x.append(os.path.join(path_image, image))
                    train_y = [os.path.join(path_image, image).replace('leftImg8bit.png', 'gtFine_labelIds.png').replace(x,y) for image
                           in sorted(os.listdir(path_image))]
            elif trainvaltest == 'val':
                val_x = [os.path.join(path_image, image) for image in sorted(os.listdir(path_image))]
                val_y = [os.path.join(path_image, image).replace('leftImg8bit.png', 'gtFine_labelIds.png').replace(x,y)for image in
                         sorted(os.listdir(path_image))]
            elif trainvaltest == 'test':
                test_x = [os.path.join(path_image, image) for image in sorted(os.listdir(path_image))]
                test_y = [os.path.join(path_image, image).replace('leftImg8bit.png', 'gtFine_labelIds.png').replace(x,y)for image in
                          sorted(os.listdir(path_image))]
    print('Done!')
    print(train_x)
    print(len(train_x))
    print(train_y)
    print(test_x)
    print(len(test_x))
    print(test_y)
    return train_x,train_y,val_x,val_y,test_x,test_y

cityscapes_dataloader()