# DATA SIOURCE:

## Autheticity of Data : 
    This data is authetic and provided by the Uprade legel source.
    We have all rights to utilize it and modify it to build the machine learning model.

**The Folder structure is as below :**

├── datasets/

│   ├── README.md

│   ├── rawdatasets/

│   │   ├── Digital images of defective and good condition tyres

|   |   |   ├──defective

|   |   |   ├──good

│   |   └── Faultfindy.zip

│   │

│   ├── random_samples_to_test_model/

│   │   ├── test1.jpg

│   │   ├── test2.jpg

│   │   ├── test 3.jpg

│   |   └── .....

│   │   

│   └── processed/

│       ├── cnn_preprocessed_test_datasets.tfrecord

│       ├── cnn_preprocessed_train_datasets.tfrecord

│       ├── cnn_preprocessed_validation_dataset.tfrecord

│       ├── resnet_X_train.pkl

│       ├── resnetX_test.pkl

│       ├── resnety_test.pkl

│       ├── resnety_train.pkl

│       ├── mobilenetV2_test_datasets.tfrecord

│       ├── mobilenetV2_validation_datasets.tfrecord

│       ├── mobilenetV2_train_datasets.tfrecord

│       ├── vgg16_test_datasets.tfrecord

│       ├── X_test.pkl

│       ├── X_train.pkl

│       ├── y_test.pkl

│       └── y_train.pkl


## rawdatasets :
    This folder have the raw data given to work on to build the model and which was in Zip file as Faultfindy.zip
    We extracted and the final data in Digital images of defective and good condition tyres
    this have two folders - Defective and Good which will be the classes of traning the model.

    ** When we executed EDA file the data will downloaded and extracted with the help of tensorflow get file util, i can't upload the data due to large volume**

## Processed data :
    The processed data is basiclly in the form of tensor object and pkl files.
    We have 5 differnt processed data
    1. CNN processed Data -  which is process and agumented for CNN models
    2. Mobilenet Data - which is processed by mobilenet libraries.
    3. VGG16 Data - which is processed by VGG16 libraries
    4. RESNET50 Data - which is processed by resnet libraries
    5. For ML algos wehave common data rescaled and processed by HOG.

    ![image](https://github.com/sonujain-qyery/faultFinding_Upgrade_Cap_Project/assets/145376236/2f543839-8924-4c53-a380-e2ead8f338e4)

    **The Tensorflow processed data size is more than git limit 100 mb due to that I can't upload the tensor processed data**



## Randome Sample Data :
    This folder have some random images downloaded from google to test the models.
