# Notebook folder structure :

├── notebooks/  
|   ├── README.md  
│   ├── CNN Model  
|   │   ├── 4.1 CNN_data_preprocessing_feature_eng.ipynb   
|   │   ├── 4.2 CNN_model_building_training.ipynb  
|   │   └── 4.3 CNN_model_evaluation.ipynb  
│   ├── DATA EDA  
|   │   └── 1. exploratory_data_analysis.ipynb  
│   ├── Data Preprocessing and Feature Engg for ML models  
|   │   └── 2. data_preprocessing_feature_engineering_ML.ipynb  
│   ├── Decision Tree Model  
|   │   ├── decision_tree_model_eval.ipynb  
|   │   └── decision_trree_model_building.ipynb  
│   ├── KNN Model  
|   │   ├── KNN_model_building.ipynb  
|   │   └── KNN_model_eval.ipynb  
│   ├── Mobilenet (TransferLayer) model  
|   │   ├── MobileNetV2_building_finetunning.ipynb  
|   │   └── MobilenetV2_model_eval.ipynb  
│   ├── Random Forest Model  
|   │   ├── RandomForestClassifier_model_building.ipynb    
|   │   ├── randomForestClassifier_model_evaluation.ipynb  
|   │   ├── rfc_hyperTuning_model_eval.ipynb    
|   │   └── rfc_hyperTuning_RandomSearchCV.ipynb  
│   ├── VGG16 Model  
|   │   ├── VGG16_building_finetunning.ipynb  
|   │   ├── VGG16_model_eval.ipynb  
|   │   ├── VGG16_FeedbackLoop.ipynb  
|   │   └── VGG16_modeltest_withrandomsamples.ipynb  
│   └── XGB Model   
|       ├── RESENT50_XGB  
|       │   ├── resnet_xgb_model_test_with_some_samples.ipynb  
|       │   ├── resnet50_xgb_datapreprocessing_feature_eng.ipynb  
|       │   ├── resnet50_xgb_evalution.ipynb  
|       │   ├── RESNET_XGB_FeedbackLoop   
|       │   └── resnet50_XGB_model_building.ipynb  
|       └── XGB classifier  
|           ├── xgb_model_building.ipynb  
|           └── xgb_model_eval.ipynb  

# Focus Area :

## Data Collection/EDA: 
    Gather historical manufacturing data, including good and faulty corresponding tyre images.
    The libraries used for EDA is : zipfile/os/numpy/tensorflow/matplotlib/seaborn

## Data Preprocessing: 
    Clean, preprocess, and transform the data to make it suitable for deep learning models.
    pickle/tensorflow.keras's layers,Model,Rescale/cv2//train_test_split/random/Data augumentation/keras.preprocessing import image

## Feature Engineering: 
    Extract relevant features and identify key process variables that impact faulty tyre generation.
    vgg16 preprocess_input/mobilenet preprocess_input/resnet preprocess_input/skimage.feature's hog

## Model Selection: 
    Choose appropriate machine learning algorithms for faulty tyre prediction.
    sklearn's DecisionTreeClassifier/KNeighborsClassifier/RandomForestClassifier/CNN/Sequential/tensorsApp(VGG16/mobilenetV2/ResNet50)/xgb

from sklearn.model_selection import 

## Model Training: 
    Train the selected models using the preprocessed data.
    Based on model fit the model.

## Model Evaluation: 
    Assess the performance of the trained models using appropriate evaluation metrics.
    accuracy_score ,classification_report,confusion_matrix,roc_auc_score,precision_score,recall_score
    f1_score,roc_curve,precision_recall_curve

## Hyperparameter Tuning:
    Optimize model hyperparameters to improve predictive accuracy.
    RandomizedSearchCV/finetuning in Deeplearning
    
**Also For the best performed model , We have included one more notebook to test model with random sample to veriy the performance.**

# Models :
To achive the best accuracy we have applied multiple models on data sets
1. CNN
2. Decision Tree
3. KNN
4. Mobilenet Pretrained model as transfer layer
5. XGB
6. Random Forest
7. VGG16 Pretrained model as transfer layer
8. RESNET50 + XGB

For Hyper Tunning :
1. RandomSerachedCV
2. Fine Tunning with Base layer tranaible

# OutComes :

| Model Name  | Accuracy |
| --------    | -------- |
| RESNET-XGB  | 93.54%   |
| VGG16       | 91.07%   |
| MobilenetV2 | 90.17%   |
| XGB         | 76.61%   |
| RandForest  | 74.73%   |
| RFC-RSCV    | 74.73%   |
| DTree       | 67.20%   |
| CNN         | 66.85%   |
| KNN         | 66.12%   |

