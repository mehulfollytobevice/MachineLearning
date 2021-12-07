:: This is the batch file for testing the random forest model on the MNIST dataset 
@echo off
title MNIST batch file 2
echo Welcome to the circus!
python train.py --fold 0 --model random_forest
python train.py --fold 1 --model random_forest
python train.py --fold 2 --model random_forest
python train.py --fold 3 --model random_forest
python train.py --fold 4 --model random_forest
pause