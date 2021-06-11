# Eff-UNet
A semantic segmentation model for Citycapes dataset that uses EfficientNet as the backbone and UNet as the decoder. This repo uses models from [Segmentation Models](https://github.com/qubvel/segmentation_models)
![image](https://user-images.githubusercontent.com/25299756/121748703-0b60fa00-cb27-11eb-864f-b4bfe1339047.png)


## Getting Started 


1. Clone the repository
```
git clone https://github.com/kodapurath/Eff-UNet/
cd Eff-UNet
conda env create -f environment.yml
```
2. Download Cityscapes dataset
    * Create an account on [Cityscapes website](https://www.cityscapes-dataset.com/login/)
    
    * Input the username and password from the website into the variables ```cityscapes_username``` and ```cityscapes_password``` inside **cityscapes_downloader.py**
```
python cityscapes_downloader.py
```
   * The directory should look like this 
   
![image](https://user-images.githubusercontent.com/25299756/121750629-60523f80-cb2a-11eb-961a-f21b6501234e.png)

3. Train the network 
```
python train.py
```

4. Inferencing the trained network

```
python infer.py
```
The backbone, decoder, activations and number of classes of output can be customized in **train.py**

```
BACKBONE = 'efficientnetb0'
model = sm.Unet(BACKBONE, classes=n_classes, activation=activation)

```
