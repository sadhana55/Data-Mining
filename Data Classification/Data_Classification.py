# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 15:14:40 2016

@author: Sadhana
"""

"""
Created on Thu Nov 10 14:50:39 2016

@author: Sadhana
"""
"""Classification model used is: K Nearest Neighbors
Step1: Read the csv file
Step2: Feature Selection: Select appropriate features 
Step3: Data munging :Selecting subset of data that is useful is predicting rather than complete data
Step4: Partition the dataset: Partition into 75% training and 25%test data 
Step5: Train the dataset: Apply the KNN with best parameters
Step6: Predict: Predict classes for test dataset
Step7: Accuracy: Calculate the accuracy
Step8: Confusion Matrix: Print the confusion Matrix
Step9: Cross Validation: Apply KNN with same parameters, use 10 fold validation
Step10: Average Cross Validation: Calculate Average of all Cross validation values
"""
#Below are the modules imported

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing


#read from the csv file and return a Pandas DataFrame.
nba = pd.read_csv('NBAstats.csv')

# headers of column names
original_headers = list(nba.columns.values)

# "Position (pos)" is the class attribute we are predicting. 
class_column = 'Pos'

"""====================Feature Importance  =================================
Below are the features that is derived from feature_importance module of 
decision tree model. I have commented this, as it dynamically provides with best features
and varies everytime. 
I have used the best combination based on the below feature importance
score and hard coded in the feature_columns_selected
    
======================Feature Importance  =================================""" 
#   
#feature_columns_all_col = ['Age', 'G', 'GS', 'MP', 'FG', 'FGA', 'FG%', '3P', '3PA', \
#    '3P%', '2P', '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', \
#    'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PS/G']
#    
#nba_feature_complete = nba[feature_columns_all_col]
#nba_class_complete = nba[class_column]
#train_feature, test_feature, train_class, test_class = \
#    train_test_split(nba_feature_complete, nba_class_complete, stratify=nba_class_complete, \
#    train_size=0.75, test_size=0.25)
#
#from sklearn.tree import DecisionTreeClassifier
#sortedfeatures = []
#
#print("----------------------------------------\n")
#print("DECISION TREE CLASSIFIER")
#tree = DecisionTreeClassifier(criterion='gini',presort=True)
#tree= tree.fit(train_feature, train_class)
#
##print("Training set score: {:.3f}".format(tree.score(train_feature, train_class)))
##print("Test set score: {:.3f}".format(tree.score(test_feature, test_class)))
##print("Test set accuracy: {:.2f}".format(tree.score(test_feature, test_class)))
#importances = tree.feature_importances_
#print ("Features sorted by their score:")
#sortedfeatures = sorted(zip(map(lambda train_feature: train_feature, tree.feature_importances_), nba_feature_complete), reverse=True)
##sorted(tree.feature_importances_,reverse=True)
#topten = []
#topten = sortedfeatures[:11]
#imp_feature = []
#imp_feature = (topten[0][1],topten[1][1],topten[2][1],topten[3][1],topten[4][1],topten[5][1],topten[6][1],topten[7][1])
#imp_feature_list = list(imp_feature)
#print (imp_feature_list)

"""==========================================================================="""

feature_columns_selected =['3PA', 'AST','TRB', 'eFG%', 'BLK', 'STL', 'FT%', 'ORB']

#Pandas DataFrame allows you to select columns. 
#We use column selection to split the data into features and class. 
nba_feature_all_selected = nba[feature_columns_selected]
nba_class_all = nba[class_column] 

"""Data Preprocessing or munging.Select few relevant data for training purpose"""
#
#nba1 = nba[(nba['MP']>7) & (nba['TRB'] < 3.5) & (nba['G']> 5) & (nba['FT%']>0.6)]
nba1 = nba[(nba['MP']>4) & (nba['G']> 4) & (nba['BLK']< 2.3)]
print(nba1)
#nba1 = nba[(nba['MP']>7) & (nba['G']> 5)]

#Apply the columns
nba_feature_processed = nba1[feature_columns_selected]
nba_class_processed = nba1[class_column]

"""Complete DataSet with selected columns is divided into Training and Test Set into 75% and 25% respectively"""
train_feature_all, test_feature_all, train_class_all, test_class_all = \
    train_test_split(nba_feature_all_selected, nba_class_all, stratify=nba_class_all, \
    train_size=0.75, test_size=0.25)

"""Preprocessing the data values using normalization"""
#nba_feature_all_norm = preprocessing.normalize(nba_feature_all_selected)
train_feature_all_norm = preprocessing.normalize(train_feature_all)
test_feature_all_norm = preprocessing.normalize(test_feature_all)

"""Dividing the processed data into Training and Test Set into 75% and 25% respectively"""
train_feature_processed, test_feature_processed, train_class_processed, test_class_processed = \
    train_test_split(nba_feature_processed, nba_class_processed, stratify=nba_class_processed, \
    train_size=0.75, test_size=0.25)

"""Normalizing the processed values that is split"""
nba_feature_processed_norm = preprocessing.normalize(nba_feature_processed)
train_feature_processed_norm = preprocessing.normalize(train_feature_processed)
test_feature_processed_norm = preprocessing.normalize(test_feature_processed)


"""===========Using KNN model to classify with the below parameters==========
These parameters are derived from Exhaustive Grid Search that I applied to 2 parameters 
to identify the best parameters
I have commented the code for Grid Search that I used for selecting the main 2 parameters
weight and n_neighbors. 
It calculates dynamically that changes sometimes and doesnot perform well in model,
However, I tested with combinations that it gave for few runs and hard coded 
the best performing parameter in the model

========================Tuning the parameters================================"""

#from sklearn.grid_search import GridSearchCV
#
#k_range = []
#k_range = list(range(1,31))
#weight_types = ['uniform', 'distance']
##print (k_range)
#param_grid = dict(n_neighbors = k_range, weights = weight_types )
##print(param_grid)
#knn=KNeighborsClassifier()
#grid =  GridSearchCV(knn,param_grid,cv=10,scoring = 'accuracy')
#grid.fit(nba_feature_all,nba_class_all)
#grid.grid_scores_
#print(grid.best_params_)
#print(grid.best_estimator_)
#print(grid.best_score_)


#knn = grid.best_estimator_
#--------------------------------------------------------------------------

"""The final best parameters hard coded"""
knn = KNeighborsClassifier(n_neighbors = 11,weights = 'distance',metric = 'minkowski', p =2)

"""Fit the Model to the training set after processing and then test it on complete dataset
and not processed or munged data"""

knn.fit(train_feature_all_norm, train_class_all)
prediction = knn.predict(test_feature_all) #predict the test features on all data and not processed data

#knn.fit(train_feature_processed_norm, train_class_processed)
#prediction = knn.predict(test_feature_all) #predict the test features on all data and not processed data

#Print the Training and Test Accuracy 
#print("Training set score: {:.3f}".format(knn.score(train_feature_norm, train_class_p)))
print("=================================================")
print("1.) Using K Nearest Neighbors")
print("=================================================")
print("2.) Test set accuracy: {:.2f}".format(knn.score(test_feature_all_norm, test_class_all)))
print("=================================================")
#Print Confustion Matrix
print("3.)Confusion matrix:")
print("-----------------------------------------")
print(pd.crosstab(test_class_all, prediction, rownames=['True'], colnames=['Predicted'], margins=True))
print("=================================================")
#    

"""10 fold Stratified Cross Validation using the cross_val_Score  """
from sklearn.model_selection import cross_val_score
from sklearn import pipeline
knn_cv = pipeline.make_pipeline(preprocessing.Normalizer(),knn)
scores = cross_val_score(knn_cv,nba_feature_all_selected,nba_class_all,cv=10,scoring='accuracy')
print("4.) Cross-validation scores using cross_val_score: {}".format(scores))
#print("=================================================")
print("5.)Average cross-validation score using cross_val_score: {:.2f}".format(scores.mean()))
print("=================================================")
