# FaultFindy

## Build intelligence using Machine Learning to predict the faulty tyre in manufacturing

The objective of this project is to develop an intelligent system using deep learning to predict faults in manufacturing processes. By analyzing various manufacturing parameters and process data, the system will predict faulty tires generated during production. This predictive capability will enable manufacturers to proactively optimize their processes, reduce waste, and improve overall production efficiency.

## Data Source

Upgrade is the data source of Fault finding of Tyres Images.

url = "https://kh3-ls-storage.s3.us-east-1.amazonaws.com/Updated%20Project%20guide%20data%20set/Faultfindy.zip"


## Installation Instructions

Below are the software should be installed on your machine to utilize this project. Use pip install command to install required libraries.

### Jupyter Notebook

The code is developed using Jupyter Notebook IDE.

### Python

Jupyter Notebook runs on Python, so you need to have Python installed on your system. You can install Python from the official Python website or via package managers like Anaconda.

### Anaconda

Anaconda is a Python distribution that comes with many pre-installed packages and tools for scientific computing and data science. It includes Jupyter Notebook, Python, and popular libraries like NumPy, pandas, and scikit-learn.

#### Modules

- **NumPy**: A fundamental package for scientific computing with Python.
- **pandas**: A powerful library for data manipulation and analysis.
- **matplotlib**: A plotting library for creating static, interactive, and animated visualizations in Python.
- **seaborn**: A statistical data visualization library built on top of matplotlib.
- **scikit-learn**: A machine learning library that provides tools for data mining and analysis.
- **TensorFlow**: An open-source machine learning framework developed by Google Brain.
- **Keras**: A high-level neural networks API that can run on top of TensorFlow, Theano, or Microsoft Cognitive Toolkit (CNTK).
- **pickle**: A Python module used for serializing and deserializing Python objects.

## Usage

The project structure is as follows:

faultyFinding/  
│  
├── README.md  
├── datasets/  
│ ├── README.md  
│ ├── rawdatasets/  
│ │ ├── Digital images of defective and good condition tyres  
│ │ │ ├── defective  
│ │ │ ├── good  
│ │ └── Faultfindy.zip  
│ ├── random_samples_to_test_model/  
│ │ ├── test1.jpg  
│ │ ├── test2.jpg  
│ │ ├── test 3.jpg  
│ │ └── .....  
│ └── processed/  
│ ├── cnn_preprocessed_test_datasets.tfrecord  
│ ├── cnn_preprocessed_train_datasets.tfrecord  
│ ├── cnn_preprocessed_validation_dataset.tfrecord  
│ ├── resnet_X_train.pkl  
│ ├── resnetX_test.pkl  
│ ├── resnety_test.pkl  
│ ├── resnety_train.pkl  
│ ├── mobilenetV2_test_datasets.tfrecord  
│ ├── mobilenetV2_validation_datasets.tfrecord  
│ ├── mobilenetV2_train_datasets.tfrecord  
│ ├── vgg16_test_datasets.tfrecord  
│ ├── X_test.pkl  
│ ├── X_train.pkl  
│ ├── y_test.pkl  
│ └── y_train.pkl  
│  
├── notebooks/  
│ ├── README.md  
│ ├── CNN Model  
│ │ ├── 4.1 CNN_data_preprocessing_feature_eng.ipynb  
│ │ ├── 4.2 CNN_model_building_training.ipynb  
│ │ └── 4.3 CNN_model_evaluation.ipynb  
│ ├── DATA EDA  
│ │ └── 1. exploratory_data_analysis.ipynb  
│ ├── Data Preprocessing and Feature Engg for ML models  
│ │ └── 2. data_preprocessing_feature_engineering_ML.ipynb  
│ ├── Decision Tree Model  
│ │ ├── decision_tree_model_eval.ipynb  
│ │ └── decision_trree_model_building.ipynb  
│ ├── KNN Model  
│ │ ├── KNN_model_building.ipynb  
│ │ └── KNN_model_eval.ipynb  
│ ├── Mobilenet (TransferLayer) model  
│ │ ├── MobileNetV2_building_finetunning.ipynb  
│ │ └── MobilenetV2_model_eval.ipynb  
│ ├── Random Forest Model  
│ │ ├── RandomForestClassifier_model_building.ipynb  
│ │ ├── randomForestClassifier_model_evaluation.ipynb  
│ │ ├── rfc_hyperTuning_model_eval.ipynb  
│ │ └── rfc_hyperTuning_RandomSearchCV.ipynb  
│ ├── VGG16 Model  
│ │ ├── VGG16_building_finetunning.ipynb  
│ │ ├── VGG16_model_eval.ipynb  
│ │ ├── VGG16_FeedbackLoop.ipynb  
│ │ └── VGG16_modeltest_withrandomsamples.ipynb  
│ └── XGB Model  
│ ├── RESENT50_XGB[**This the final model considered due to high performance accuracy**]   
│ │ ├── resnet50_xgb_model_test_with_some_samples.ipynb  
│ │ ├── resnet50_xgb_datapreprocessing_feature_eng.ipynb  
│ │ ├── resnet50_xgb_evalution.ipynb  
│ │ ├── resnet50_xgb_FeedbackLoop.ipynb  
│ │ └── resnet50_XGB_model_building.ipynb  
│ └── XGB classifier  
│ ├── xgb_model_building.ipynb  
│ └── xgb_model_eval.ipynb  
│  
├── model/  
│ └── model/  
│ ├── cnn_model.keras  
│ ├── decisionTreeModel.pkl  
│ ├── knn_model.pkl  
│ ├── mobilenet_model.keras  
│ ├── randomForestClassifier.pkl  
│ ├── RESNET50_xgbClassifier_model.pkl  
│ ├── rfc_hyperTuningRandomSerachCV.pkl  
│ ├── vgg16_model.keras  
│ └── xgbClassifier.pkl  
│
└── visualizations/  
├── class_distribution.png  
├── CNN_accuracy_plot.png  
├── CNN_loss_plot.png  
├── CNN_loss_plot.png  
├── .......  
└── xgb_ROC_Curve.png  
