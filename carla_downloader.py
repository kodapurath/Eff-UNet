import os
from glob import glob

def data_path_loader(dataset_path='/content/drive/MyDrive/Ottonomy/Carla_images'):
    
    def image_accumulator_from_folder(list_of_folders):
        # print(list_of_folders)
        image_paths = []
        print(list_of_folders)
        #if "RGB" in list_of_folders[0]:
        for folder in list_of_folders:
	        for image in sorted(glob(os.path.join(folder + "/*.png"))):
	          image_paths.append(image)
#else:
         #   for folder in list_of_folders:
          #      for image in sorted(glob(os.path.join(folder + "/*"))):
           #         image_paths.append(image)
        # print("IMAGE PATHS :", image_paths)
        print("Folders  : ", len(list_of_folders))
        print("Images : ", len(image_paths))
        return image_paths

    train_x_place_folders = sorted(glob(os.path.join(dataset_path, "RGB/Train/*")))
    # print(train_x_place_folders)
    # train_x_place_folders = sorted(glob(os.path.join(dataset_path, "leftImg8bit/train/*")))

    # print(train_x_place_folders)
    train_y_place_folders = sorted(glob(os.path.join(dataset_path, "Sem/Train/*")))
    # train_y_place_folders = sorted(glob(os.path.join(dataset_path, "gtFine/train/*")))

    # print(train_y_place_folders)
    print("***********************")
    print("----------------------")

    print("Train X")
    print("----------------------")
    trainX_paths = image_accumulator_from_folder(train_x_place_folders)
    print("----------------------")
    print("Train Y")
    print("----------------------")
    trainY_paths = image_accumulator_from_folder(train_y_place_folders)

    # test_x_place_folders = sorted(glob(os.path.join(dataset_path, "leftImg8bit/test/*")))
    test_x_place_folders = sorted(glob(os.path.join(dataset_path, "RGB/Test/*")))

    # print(test_x_place_folders)
    # test_y_place_folders = sorted(glob(os.path.join(dataset_path, "gtFine/test/*")))
    test_y_place_folders = sorted(glob(os.path.join(dataset_path, "Sem/Test/*")))

    # print(test_y_place_folders)
    print("***********************")
    print("----------------------")

    print("Test X")
    print("----------------------")
    testX_paths = image_accumulator_from_folder(test_x_place_folders)
    print("----------------------")
    print("Test Y")
    print("----------------------")
    testY_paths = image_accumulator_from_folder(test_y_place_folders)

    val_x_place_folders = sorted(glob(os.path.join(dataset_path, "RGB/Val/*")))

    # print(test_x_place_folders)
    # test_y_place_folders = sorted(glob(os.path.join(dataset_path, "gtFine/test/*")))
    val_y_place_folders = sorted(glob(os.path.join(dataset_path, "Sem/Val/*")))

    # print(test_y_place_folders)
    print("***********************")
    print("----------------------")

    print("Val X")
    print("----------------------")
    valX_paths = image_accumulator_from_folder(val_x_place_folders)
    print("----------------------")
    print("Val Y")
    print("----------------------")
    valY_paths = image_accumulator_from_folder(val_y_place_folders)


    return trainX_paths, trainY_paths, valX_paths, valY_paths, testX_paths, testY_paths


if __name__ == '__main__':
    trainX_paths, trainY_paths, valX_paths, valY_paths, testX_paths, testY_paths = data_path_loader()
    print(trainX_paths)
    print(trainY_paths)
    # print(len(trainX_paths))

