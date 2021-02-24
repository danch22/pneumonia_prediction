# Pneumonia Predictor

Convolutional Neural Network for predicting pneumonia in patients by classifying X-Ray images of their chests. We are using pre-trained MobileNet weights for the CNN.
The model will predict whether a person is healthy or sick and it will output a visual explanation of the decision made. 

For example:

#### Healthy patient
Here our model predicted that the patient is healty and the explainability model shows why. 
It will show in red all the areas that are evidence of sickness and in green the evidence of healthiness. Here we can see that the patient is healthy.

<img src="https://raw.githubusercontent.com/ygoldfrid/pneumonia_prediction/master/images/pred_test_image_1.jpeg" height="250">

#### Sick patient
Similar for a sick patient. Here we can understand that the evidence of sickness is greater because the red areas are much larger than the green.

<img src="https://raw.githubusercontent.com/ygoldfrid/pneumonia_prediction/master/images/pred_test_image_2.jpeg" height="250">

# Usage

#### Prerequisites

Due to Tensorflow requirements, this project will only work wiht Python versions `3.6` or `3.7` so if you have an older or newer version you'll have to install one of the supported ones. 
By the way, you can have multiple versions of Python installed in your machine (I have 3 different ones myself!)

#### 1. Create and activate Virtual Environment (optional but highly recommended)
Windows:
```
python -m venv env
env\Scripts\activate.bat
```
Mac/Linux:
```
python3 -m venv env
source env/bin/activate
```

#### 2. Install requirements 
It might take a while, but it's worth it
```
pip install -r requirements.txt
```

#### 3. Execute
| If executed locally, you might see a lot of Tensorflow warnings, don't worry that's probably because you are not using a GPU |
| --- |


```
python server.py
```

# Dataset
Kaggle's [Chest X-Ray Images](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) dataset compiled by [Paul Mooney](https://www.kaggle.com/paultimothymooney)

There are 5,863 X-Ray images (JPEG) and 2 categories (Pneumonia/Normal).

Chest X-ray images (anterior-posterior) were selected from retrospective cohorts of pediatric patients of one to five years old from Guangzhou Women and Children’s Medical Center, Guangzhou. All chest X-ray imaging was performed as part of patients’ routine clinical care.

For the analysis of chest x-ray images, all chest radiographs were initially screened for quality control by removing all low quality or unreadable scans. The diagnoses for the images were then graded by two expert physicians before being cleared for training the AI system. In order to account for any grading errors, the evaluation set was also checked by a third expert.

# Authors

This project, was made by:
* [Dana Velibekov](https://github.com/danch22)
* [Yair Stemmer](https://github.com/yairstemmer)
* [Yaniv Goldfrid](https://github.com/ygoldfrid)
