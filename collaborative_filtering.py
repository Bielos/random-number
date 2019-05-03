# -*- coding: utf-8 -*-
"""
Created on Thu May  2 21:50:25 2019

@author: Bielos
"""
#Imports
import numpy as np
import pandas as pd
import EDA_framework as EDA
import surprise

#Data cleaning
df = pd.read_csv('random-data(1).csv')
df['DATE'] = pd.to_datetime(df['DATE'])

df['NUMBER'] = df['NUMBER'].apply(lambda x: np.NaN if x == 'xx' or x == '**' else x)
df = EDA.delete_null_observations(df, 'NUMBER')

#Surprises' dataset creation
surprise_dataset = pd.DataFrame()
surprise_dataset['userID'] = np.ones(len(df))
surprise_dataset['itemID'] = np.ones(len(df))
surprise_dataset['rating'] = pd.to_numeric(df['NUMBER']).values

reader = surprise.Reader()
data = surprise.Dataset.load_from_df(surprise_dataset, reader)

#Algorithm Evaluation
knn = surprise.prediction_algorithms.knns.KNNBasic()
svd = surprise.prediction_algorithms.matrix_factorization.SVD()
normal = surprise.prediction_algorithms.random_pred.NormalPredictor()

print('--------------------- K-NN ---------------------')
from surprise.model_selection import cross_validate
cross_validate(knn, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

print('--------------------- SVD ---------------------')
from surprise.model_selection import cross_validate
cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

print('--------------------- NORMAL PREDICTOR ---------------------')
from surprise.model_selection import cross_validate
cross_validate(normal, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)