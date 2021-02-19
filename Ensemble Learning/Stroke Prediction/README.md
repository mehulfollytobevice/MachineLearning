# Stroke Prediction using Stacking and Blending
According to the World Health Organization (WHO),stroke is the 2nd leading cause of death globally, responsible for approximately 11% of total deaths.
This project is aimed at creating an effective learning system to predict strokes so that pre-emptive action can be taken with regards to a patient's treatment.<br>

This folder contains several jupyter notebooks which contain different parts of the machine learning pipeline . The file descriptions are given below:<br>
1. <b>StrokePrediction_Data_Extraction_and_EDA.ipynb :</b> In this notebook , we download the required dataset from kaggle and perform extensive EDA on the data .
2. <b>StrokePrediction_Data_Preprocessing_And_Training.ipynb: </b> In this notebook,we preprocess the data and create the models to classify positive and negative cases. We implement stacking and blending to make better models and improve the predictive power.
3. <b>StrokePrediction_HPO_and_Evaluation.ipynb :</b> In this notebook , we perform hyperparameter optimization using several methods and tune our model.Then we evaluate the optimized model using ROC and AUC.
4. <b>StrokePrediction_Model_Explaination.ipynb:</b> In this notebook , we try to understand how our model works using measures like permutation importance and SHAP values .

The dataset used in this project can be found [here](https://www.kaggle.com/fedesoriano/stroke-prediction-dataset).
