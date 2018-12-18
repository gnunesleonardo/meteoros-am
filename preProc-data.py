import csv
import pandas as pd
from pandas import *
from sklearn import preprocessing
import numpy as np

import os
os.chdir('/home/sramarocas/unifesp/meteoros-am')

######################################################################
## Conjunto de treino ##
######################################################################
## importa conjunto de treino ##
train_data = pd.read_csv('train_raw.csv')
train_data.shape

## retira coluna 'Id' ##
train_data = train_data.drop('Id', axis=1)

## retira coluna 'Class' ##
train_data = train_data.drop('Class', axis=1)

## verifica a existenia de NAN no conjunto de treino ##
np.sum(np.sum(np.isnan(train_data)))
## retira NANs presentes ##
train_data = train_data.fillna(0)

######################################################################

######################################################################
## Conjunto de teste ##
######################################################################
## importa o conjunto de teste ##
test_data = pd.read_csv('test_raw.csv')
test_data.shape

## retira coluna 'Id' ##
test_data = test_data.drop('Id', axis=1)
## verifica a existenia de NAN no conjunto de treino ##
np.sum(np.sum(np.isnan(test_data)))
## retira NANs presentes ##
test_data = test_data.fillna(0)

######################################################################

######################################################################
## Gabarito da classificacao ##
######################################################################
## importa o conjunto de classificacao ##
class_data = pd.read_csv('classification.csv')

## separa classificacoes publicas e privadas ##
priv_class = class_data[class_data.Usage == 'Private']
publ_class = class_data[class_data.Usage == 'Public']

priv_class.to_csv('priv_class.csv', index=False)
publ_class.to_csv('publ_class.csv', index=False)

######################################################################

######################################################################
## Removendo atributos com baixa variacao ##
######################################################################
from sklearn.feature_selection import VarianceThreshold

## Remove os atributos que a variacao e zero ##
aux = train_data
var_zero = VarianceThreshold()
model = var_zero.fit(aux)

train_data = train_data.loc[:, model.get_support()]
test_data = test_data.loc[:, model.get_support()]

train_data.shape
test_data.shape

train_data.to_csv('train_df.csv', index=False)
test_data.to_csv('test_df.csv', index=False)

######################################################################
