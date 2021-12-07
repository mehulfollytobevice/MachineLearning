:: This is the batch file for testing different models on the MNIST dataset
@echo off
title MNIST batch file
echo Welcome to the circus!
python train.py --fold 0 --model decision_tree_gini
python train.py --fold 0 --model decision_tree_entropy
python train.py --fold 0 --model random_forest
pause