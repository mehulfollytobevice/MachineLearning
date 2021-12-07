# MNIST using Random Forest 
In this micro-project, we create a basic machine learning framework and test it using the MNIST dataset. The structure of this project is inspired from Abhishek Thakur's book "Approaching Almost Any Machine Learning Problem". The book is open-source and you can download it from [here](https://github.com/abhishekkrthakur/approachingalmost). 

### Which algorithms are used here?
* Decision Trees
* Random Forests 

### What is the structure of the project?
1. <b> Input folder: </b> This folder consists of all the input files and data for your machine learning project.
2. <b> Src folder: </b> All the python scripts associated with the project are kept here.
    * create_folds.py: This file contains the code for downloading the MNIST dataset and splitting it into multiple folds which can be used for training and validation purposes.  
    * train.py: This file contains the code for training and evaluating the model on the dataset.
    * config.py: This file contains the information like the fold numbers, the training file and the output folder. While using this project, please change the paths mentioned in this file according to your system.
    * model.dispatcher.py: Within this file, we define a dictionary with keys that are names of the models and values are the models themselves.
3. <b> Models folder: </b> This folder keeps all the trained models.

### How to test the project?
1. Navigate to the '/src' folder.
2. Locate the batch files (train_batch_file.bat and train_batch_rf.bat).
3. If you are using Windows, simply click on the batch files to execute them. Otherwise, use the commands in the batch files and customize them according to your operating system.
4. On execution, you will see the output of the experiments showing the model name, the corresponding fold number and the accuracy achieved.



